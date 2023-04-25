import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import pickle

import gensim

from torch.nn.utils.rnn import pad_sequence
from typing import Any
from typing import Dict, List

import gensim.downloader
from gensim.models import KeyedVectors


SEED = 1234

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates

    model = StudentModel(device)
    return model


class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]

class EventDetectionDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


with open("model/vocabulary.pickle", "rb") as file:
    vocabulary = pickle.load(file)

with open("model/label_vocabulary.pickle", "rb") as file:
    label_vocabulary = pickle.load(file)


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

from numpy.lib.arraypad import pad
#method that adds paddding to sequences and returns tensors dictionary
def prepare_batch(batch: List[List[str]]) -> List[Dict]:
  # extract features from batch
  x = [sentence for sentence in batch]

  # convert tokens to index 
  x = [[vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in sentence] for sentence in x]

  # convert features to tensor and pad them
  x = pad_sequence(
    [torch.as_tensor(sentence) for sentence in x],
    batch_first=True,
    padding_value=vocabulary[PAD_TOKEN]
  )

  return {"x": x}

def load_torch_embedding_layer(weights: KeyedVectors, padding_idx: int = 0, freeze: bool = False):
    vectors = weights.vectors
    # random vector for pad
    pad = np.random.rand(1, vectors.shape[1])
    print(pad.shape)
    # mean vector for unknowns
    unk = np.mean(vectors, axis=0, keepdims=True)
    print(unk.shape)
    # concatenate pad and unk vectors on top of pre-trained weights
    vectors = np.concatenate((pad, unk, vectors))
    # convert to pytorch tensor
    vectors = torch.FloatTensor(vectors)
    # and return the embedding layer
    return torch.nn.Embedding.from_pretrained(vectors, padding_idx=padding_idx, freeze=freeze)

class EventDetectionModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams, word_weights):
        super(EventDetectionModel, self).__init__()
        # Embedding layer: a matrix vocab_size x embedding_dim where each index 
        # correspond to a word in the vocabulary and the i-th row corresponds to 
        # a latent representation of the i-th word in the vocabulary.
        self.word_embedding = load_torch_embedding_layer(word_weights)
        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs 
        # a new **contextual** representation of each word that depend
        # on the preciding words.
        self.lstm = nn.LSTM(hparams["word_embedding_dim"], hparams["hidden_dim"], 
                            bidirectional=hparams["bidirectional"],
                            num_layers=hparams["num_layers"], 
                            dropout = hparams["dropout"] if hparams["num_layers"] > 1 else 0)
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        lstm_output_dim = hparams["hidden_dim"] if hparams["bidirectional"] is False else hparams["hidden_dim"] * 2

        # During training, randomly zeroes some of the elements of the 
        # input tensor with probability hparams.dropout using samples 
        # from a Bernoulli distribution. Each channel will be zeroed out 
        # independently on every forward call.
        # This has proven to be an effective technique for regularization and 
        # preventing the co-adaptation of neurons
        self.dropout = nn.Dropout(hparams["dropout"])
        self.classifier = nn.Linear(lstm_output_dim, hparams["num_classes"])

    
    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    
    def __init__(self, device):
        super(StudentModel, self).__init__()
        self.device = device
        
        with open("model/index_to_label_vocab.pickle", "rb") as file:
            self.index_to_label_vocab = pickle.load(file)
        
        with open("model/best_params.pickle", "rb") as file:
            self.hparams = pickle.load(file)

        self.vocabulary = vocabulary

        self.label_vocabulary = label_vocabulary

        with open("model/"+self.hparams["word_embeddings"]+".pickle", "rb") as file:
            self.weights = pickle.load(file)

        self.model = EventDetectionModel(self.hparams, self.weights).to(device)



    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        testset = EventDetectionDataset(tokens)

        # data loader parameters
        collate_fn = prepare_batch

        test_dataloader = DataLoader(testset, collate_fn=collate_fn, batch_size=self.hparams["batch_size"], shuffle=False)

        self.model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))

        total_predictions = []
        for batch in test_dataloader:
            inputs = batch["x"].to(self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(inputs.long())
                predictions = torch.argmax(logits, -1)
                prediction_list = predictions.tolist()
                total_predictions.extend(prediction_list)
        
        total_predictions_tokens = [[self.index_to_label_vocab[label] for label in sample] for sample in total_predictions]
        
        #remove the padding, in order to make the evaluation possible
        final_predictions = []
        for i in range(len(total_predictions_tokens)):
            ith_prediction = total_predictions_tokens[i]
            predictions_without_padding = ith_prediction[:len(tokens[i])]
            final_predictions.append(predictions_without_padding)

        return final_predictions
