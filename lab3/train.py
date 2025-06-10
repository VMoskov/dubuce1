from utils import Vocab, NLPDataset, get_frequencies, pad_collate_fn
import torch
from torch.utils.data import DataLoader
from models import BaselineModel, EmbeddingLayer, VanilaRNN, GRU, LSTM
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random
from argparse import ArgumentParser


# default configuration
config = {
    'seed': random.randint(0, 1000000),
    'embedding_build_type': 'pretrained',
    'model_type': 'baseline',
    'embedding_dim': 300,
    'hidden_dim': 150,
    'num_layers': 2,
    'train_batch_size': 10,
    'test_batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 1e-4
}


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        print(f'Using device: {self.device}')

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.best_model_state_dict = None
        self.best_f1 = -1.0
        self.best_epoch = 0

    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0

        for texts, labels, lengths in train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(texts, lengths)

            loss = self.criterion(outputs.squeeze(), labels.float())
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

        return epoch_loss / len(train_loader)
    
    def train(self, train_loader, val_loader, num_epochs=10):
        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(train_loader)

            accuracy, f1, _ = self.evaluate(val_loader)

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_model_state_dict = self.model.state_dict()
                self.best_epoch = epoch + 1

            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Epoch Loss: {epoch_loss:.4f}, '
                  f'Validation Accuracy: {accuracy:.4f}, '
                  f'Validation F1: {f1:.4f}, '
                  f'Best F1: {self.best_f1:.4f} at Epoch {self.best_epoch}')

        self.model.load_state_dict(self.best_model_state_dict)
        return self.model
    
    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for texts, labels, lengths in data_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = self.model(texts, lengths)
                preds = torch.sigmoid(outputs).squeeze().round().long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        return accuracy, f1, cm
    
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
        
    @staticmethod
    def build_model(embedding_layer, config):
        model_type = config['model_type']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config['dropout']
        bidirectional = config['bidirectional']
        attention = config['attention']
        if model_type == 'baseline':
            return BaselineModel(embedding_layer)
        elif model_type == 'vanilla_rnn':
            return VanilaRNN(embedding_layer, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, attention=attention)
        elif model_type == 'gru':
            return GRU(embedding_layer, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, attention=attention)
        elif model_type == 'lstm':
            return LSTM(embedding_layer, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, attention=attention)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        

def parse_args():
    parser = ArgumentParser(description='Train a sentiment analysis model on SST dataset')
    parser.add_argument('--embedding_build_type', type=str, default=config['embedding_build_type'], choices=['random', 'pretrained'], help='Type of embedding to build')
    parser.add_argument('--vocab_size', type=int, default=-1, help='Size of the vocabulary (default: -1 for no limit)')
    parser.add_argument('--batch_size_train', type=int, default=10, help='Batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing (default: 32)')
    parser.add_argument('--model_type', type=str, default=config['model_type'], choices=['baseline', 'vanilla_rnn', 'gru', 'lstm'], help='Type of model to use')
    parser.add_argument('--embedding_dim', type=int, default=config['embedding_dim'], help='Dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=config['hidden_dim'], help='Dimension of the hidden layer')
    parser.add_argument('--num_layers', type=int, default=config['num_layers'], help='Number of layers in the RNN')
    parser.add_argument('--train_batch_size', type=int, default=config['train_batch_size'], help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=config['test_batch_size'], help='Batch size for testing')
    parser.add_argument('--num_epochs', type=int, default=config['num_epochs'], help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'], help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the RNN layers')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNNs')
    parser.add_argument('--attention', action='store_true', help='Use attention mechanism in the model')
    args = parser.parse_args()
    return args


def build_config(args):
    config['seed'] = random.randint(0, 1000000)
    config['embedding_build_type'] = args.embedding_build_type
    config['vocab_size'] = args.vocab_size if args.vocab_size > 0 else -1
    config['batch_size_train'] = args.train_batch_size
    config['batch_size_test'] = args.test_batch_size
    config['model_type'] = args.model_type
    config['embedding_dim'] = args.embedding_dim
    config['hidden_dim'] = args.hidden_dim
    config['num_layers'] = args.num_layers
    config['train_batch_size'] = args.train_batch_size
    config['test_batch_size'] = args.test_batch_size
    config['num_epochs'] = args.num_epochs
    config['learning_rate'] = args.learning_rate
    config['dropout'] = args.dropout
    config['bidirectional'] = args.bidirectional
    config['attention'] = args.attention
    return config


if __name__ == '__main__':
    args = parse_args()
    config = build_config(args)

    torch.manual_seed(config['seed'])
    print(f'Config: {config}')

    device = Trainer.get_device()

    frequencies = get_frequencies('train')

    x_vocab = Vocab(frequencies, max_size=config['vocab_size'], min_freq=1)
    y_vocab = Vocab({'positive': 1, 'negative': 0}, max_size=config['vocab_size'], min_freq=1, vocab_type='label')

    trainset = NLPDataset('train', x_vocab, y_vocab)
    valset = NLPDataset('val', x_vocab, y_vocab)
    testset = NLPDataset('test', x_vocab, y_vocab)

    train_loader = DataLoader(trainset, batch_size=config['batch_size_train'], shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(valset, batch_size=config['batch_size_test'], shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(testset, batch_size=config['batch_size_test'], shuffle=False, collate_fn=pad_collate_fn)

    embedding_layer = EmbeddingLayer(x_vocab, embedding_dim=config['embedding_dim'], build_type=config['embedding_build_type'])
    model = Trainer.build_model(embedding_layer, config=config)
    print(f'Model type: {config["model_type"]}, Embedding build type: {config["embedding_build_type"]}')

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    trainer = Trainer(model, criterion, optimizer, device)
    model = trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'])
    acc, f1, cm = trainer.evaluate(test_loader)
    print(f'Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}')
