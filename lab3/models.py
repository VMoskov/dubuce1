import torch
import torch.nn as nn
from pathlib import Path
import einops
from utils import PAD, UNK
import torch.nn.utils.rnn as rnn_utils


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super(Rearrange, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return einops.rearrange(x, self.pattern)

    def __repr__(self):
        return f'Rearrange({self.pattern})'


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab, embedding_dim=300, build_type='random'):
        super(EmbeddingLayer, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embedding_matrix = self._build_embedding_matrix(vocab, build_type, embedding_dim)

    def forward(self, x):
        embeddings = self.embedding_matrix(x)
        return embeddings

    def _build_embedding_matrix(self, vocab, build_type='random', embedding_dim=300):
        embedding_matrix = torch.randn(len(vocab.stoi), embedding_dim)
        embedding_matrix[vocab.stoi[PAD]] = 0.0  # padding vector set to zero

        if build_type == 'random':
            pretrained = False
            pass

        elif build_type == 'pretrained':
            pretrained = True
            glove_path = Path(__file__).parent / 'data' / 'sst_glove_6b_300d.txt'
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    word = parts[0]
                    if word in vocab.stoi:
                        embedding = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                        embedding_matrix[vocab.stoi[word]] = embedding
                    else:
                        # print(f"Word '{word}' not found in vocabulary, skipping.")
                        pass

        embedding_matrix = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=vocab.stoi[PAD]
        )
        return embedding_matrix
    

class Attention(nn.Module):
    '''
    Bahdanau Attention mechanism.
    '''
    def __init__(self, rnn_hidden_size):
        super(Attention, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.attention_dim = rnn_hidden_size // 2
        self.W1 = nn.Linear(rnn_hidden_size, self.attention_dim, bias=False)
        self.w2 = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, hidden_states, lengths):
        a = self.w2(torch.tanh(self.W1(hidden_states)))
        a = einops.rearrange(a, 'b l 1 -> b l')  # (batch_size, seq_length)

        max_length = hidden_states.size(1)
        indices = torch.arange(max_length, device=lengths.device).expand(len(lengths), max_length)
        mask = indices < lengths.unsqueeze(1)

        a = a.masked_fill(~mask, float('-inf'))  # apply mask to attention scores

        scores = torch.softmax(a, dim=1)
        out = torch.sum(scores.unsqueeze(-1) * hidden_states, dim=1)
        return out


class BaselineModel(nn.Module):
    def __init__(self, embedding_head):
        super(BaselineModel, self).__init__()

        self.embedding_head = embedding_head
        embedding_dim = embedding_head.embedding_dim
        self.layers = nn.Sequential(
            self.embedding_head,
            Rearrange('b l d -> b d l'),  # rearranging to (batch_size, seq_length, embedding_dim)
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=embedding_dim, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=1)
            # we dont use nn.Sigmoid() here because we use BCEWithLogitsLoss
        )

    def forward(self, x, lengths=None):  # lengths is not used in this model
        logits = self.layers(x)
        return logits
    

# --- RNN Models ---
class RNNBase(nn.Module):
    def __init__(self, embedding_head, hidden_size=150, num_layers=2, bidirectional=False, attention=False):
        super(RNNBase, self).__init__()
        self.embedding_head = embedding_head
        self.embedding_dim = embedding_head.embedding_dim

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bidirectional = bidirectional
        rnn_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        self.attention_head = Attention(rnn_output_size) if attention else None

        self.time_first = Rearrange('b l d -> l b d')
        self.batch_first = Rearrange('l b d -> b l d')

        self.rnn_module = None

        self.decoder = nn.Sequential(
            nn.Linear(rnn_output_size, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )

    def forward(self, x, lengths):
        embeddings = self.embedding_head(x)
        embeddings = self.time_first(embeddings)  # rearranging to (seq_length, batch_size, embedding_dim)

        packed_embeddings = rnn_utils.pack_padded_sequence(
            embeddings, lengths=lengths, batch_first=False, enforce_sorted=False
        )
        
        outputs, hidden = self.rnn_module(packed_embeddings)

        if self.attention_head is not None:
            rnn_outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=False)  # pad outputs
            rnn_outputs = self.batch_first(rnn_outputs)  # rearranging to (batch_size, seq_length, hidden_size)
            context = self.attention_head(rnn_outputs, lengths)  # get context vector from attention
            logits = self.decoder(context)
        else:
            last_hidden_state = self._extract_hidden_state(hidden)
            logits = self.decoder(last_hidden_state)
        return logits

    def _extract_hidden_state(self, hidden):
        pass


class VanilaRNN(RNNBase):
    def __init__(self, embedding_head, hidden_size=150, num_layers=2, dropout=0.0, bidirectional=False, attention=False):
        super(VanilaRNN, self).__init__(embedding_head, hidden_size, num_layers, bidirectional, attention)
    
        self.rnn_module = nn.RNN(input_size=self.embedding_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=False,
                                dropout=dropout if num_layers > 1 else 0.0,
                                bidirectional=bidirectional)
        
    def _extract_hidden_state(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
            return hidden
        else:
            return hidden[-1]


class GRU(RNNBase):
    def __init__(self, embedding_head, hidden_size=150, num_layers=2, dropout=0.0, bidirectional=False, attention=False):
        super(GRU, self).__init__(embedding_head, hidden_size, num_layers, bidirectional, attention)

        self.rnn_module = nn.GRU(input_size=self.embedding_dim,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=False,
                                 dropout=dropout if num_layers > 1 else 0.0,
                                 bidirectional=bidirectional)

    def _extract_hidden_state(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
            return hidden
        else:
            return hidden[-1]


class LSTM(RNNBase):
    def __init__(self, embedding_head, hidden_size=150, num_layers=2, dropout=0.0, bidirectional=False, attention=False):
        super(LSTM, self).__init__(embedding_head, hidden_size, num_layers, bidirectional, attention)

        self.rnn_module = nn.LSTM(input_size=self.embedding_dim,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=False,
                                  dropout=dropout if num_layers > 1 else 0.0,
                                  bidirectional=bidirectional)

    def _extract_hidden_state(self, hidden):
        hidden, cell = hidden
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
            return hidden
        else:
            return hidden[-1]
    