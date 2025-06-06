from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

PAD = '<PAD>'
UNK = '<UNK>'


@dataclass
class Instance:
    text: str
    label: str


class NLPDataset(Dataset):
    def __init__(self, subset, x_vocab, y_vocab):
        dir_path = Path(__file__).parent / 'data'
        file = dir_path / f'{subset}.csv'
        super().__init__()
        self.instances = self._build_instances(file)
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        text = self.instances[index].text
        label = self.instances[index].label

        text = self.x_vocab.encode(text)
        label = self.y_vocab.encode([label])[0]
        return text, label
    
    def _build_instances(self, file):
        instances = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                text, label = line.strip().split(', ')
                instances.append(Instance(text=text.split(), label=label))
        return instances


class Vocab:
    def __init__(self, frequencies, max_size, min_freq, vocab_type='text'):
        self._build_vocab(frequencies, max_size, min_freq, vocab_type)

    def _build_vocab(self, frequencies, max_size, min_freq, vocab_type):
        if vocab_type == 'label':
            frequencies = {k: v for k, v in frequencies.items() if v >= min_freq}
            frequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
            self.stoi = {word: i for i, (word, _) in enumerate(frequencies)}
            self.stoi[UNK] = len(self.stoi)  # Add UNK token to the end to avoid errors
            self.itos = {i: word for word, i in self.stoi.items()}
            return
        self.stoi = {PAD: 0, UNK: 1}

        frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))  # sort by frequency, descending
        for word, freq in frequencies.items():
            if max_size != -1:
                if len(self.stoi) >= max_size:
                    break
                if freq < min_freq:
                    continue
            self.stoi[word] = len(self.stoi)
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(word, self.stoi[UNK]) for word in text]
    

def build_embedding_matrix(vocab, build_type='random', embedding_dim=300):
    embedding_matrix = torch.randn(len(vocab.stoi), embedding_dim)
    embedding_matrix[vocab.stoi[PAD]] = 0.0  # padding vector set to zero

    if build_type == 'random':
        pretrained = False
        pass

    elif build_type == 'pretrained':
        pretrained = True
        glove_path = 'sst_glove_6b_300d.txt'
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                if word in vocab.stoi:
                    embedding = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                    embedding_matrix[vocab.stoi[word]] = embedding

    embedding_matrix = nn.Embedding.from_pretrained(
        embedding_matrix, freeze=pretrained, padding_idx=vocab.stoi[PAD]
    )
    return embedding_matrix


def get_frequencies(subset):
    dir_path = Path(__file__).parent / 'data'
    file = dir_path / f'{subset}.csv'
    frequencies = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            text, _ = line.strip().split(',')
            for word in text.split():
                frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies


def collate_fn(batch):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """
    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts])
    return texts, labels, lengths


def pad_collate_fn(batch, pad_index=0):
    texts, labels, lengths = collate_fn(batch)
    max_length = max(lengths)
    padded_texts = nn.utils.rnn.pad_sequence(
        [torch.tensor(text) for text in texts],
        batch_first=True,
        padding_value=pad_index
    )
    padded_labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, padded_labels, lengths