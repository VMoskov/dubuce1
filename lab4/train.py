import time
import einops
from collections import defaultdict
import numpy as np
import torch
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleMetricEmbedding, IdentityModel
from utils import visualize_embeddings
import matplotlib.pyplot as plt


MNIST_DOWNLOAD_ROOT = './mnist/'
BATCH_SIZE = 64
EMB_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 3
NUM_CLASSES = 10
EVAL_ON_TEST = True
EVAL_ON_TRAIN = False
REMOVE_CLASS = True


class Trainer:
    def __init__(self, model, optimizer, train_loader, test_loader, traineval_loader, device, num_classes, emb_size):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        print(f'Using device: {device}')

        self.model.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.traineval_loader = traineval_loader
        self.device = device
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.PRINT_LOSS_N = 100

        self.best_model_state_dict = None
        self.best_acc = -1.0
        self.best_epoch = 0

    def _train_epoch(self):
        self.model.train()
        losses = []
        for i, data in enumerate(self.train_loader):
            anchor, positive, negative, _ = data
            self.optimizer.zero_grad()
            loss = self.model.loss(anchor.to(self.device), positive.to(self.device), negative.to(self.device))
            loss.backward()
            self.optimizer.step()
            losses.append(loss.cpu().item())
            if i % self.PRINT_LOSS_N == 0:
                print(f'  Iter: {i}, Mean Loss: {np.mean(losses):.3f}')
        return np.mean(losses)

    def _compute_representations(self, loader):
        self.model.eval()
        representations = defaultdict(list)
        for i, data in enumerate(loader):
            anchor, identity = data[0], data[-1]
            with torch.no_grad():
                repr_batch = self.model.get_features(anchor.to(self.device))
            for i in range(identity.shape[0]):
                representations[identity[i].item()].append(repr_batch[i])

        averaged_repr = torch.zeros(self.num_classes, self.emb_size).to(self.device)
        for k, items in representations.items():
            if items:
                r = torch.stack(items).mean(0)
                norm = torch.linalg.vector_norm(r)
                if norm > 0:
                    averaged_repr[k] = r / norm
        return averaged_repr

    @staticmethod
    def _make_predictions(class_representations, query_embedding):
        '''Calculates L2 distance to find the closest class representation.'''
        # class_representations: (num_classes, emb_size)
        # query_embedding: (batch_size, emb_size)
        query_expanded = einops.rearrange(query_embedding, 'b c -> b 1 c')  # prepare for broadcasting
        diff = class_representations - query_expanded
        return (diff ** 2).sum(2)  # L2 distance for each class representation
    
    def _evaluate(self, class_representations, loader):
        '''Evaluates the model on a given data loader and returns top-1 accuracy.'''
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                anchor, identity = data
                identity = identity.to(self.device)
                with torch.no_grad():
                    query_embedding = self.model.get_features(anchor.to(self.device))
                    query_embedding = query_embedding / torch.linalg.vector_norm(query_embedding) # Normalize

                pred = self._make_predictions(class_representations, query_embedding)
                top1 = pred.min(1)[1]  # Get the index of the closest class representation
                correct += top1.eq(identity).sum().item()
                total += anchor.size(0)
        return correct / total if total > 0 else 0
    
    def _is_model_trainable(self):
        """Check if the model has trainable parameters."""
        num_trainable_elements = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return num_trainable_elements > 0

    def train(self, epochs, eval_on_test=True, eval_on_train=False, representation_loader=None):
        print(f'Training {self.model.__class__.__name__}')
        is_trainable = self._is_model_trainable()
        effective_epochs = epochs if is_trainable else 1

        for epoch in range(effective_epochs):
            print(f'--- Epoch: {epoch} ---')
            start_time = time.time()

            train_loss = self._train_epoch()
            print(f'Mean Loss in Epoch {epoch}: {train_loss:.3f}')

            if eval_on_test or eval_on_train:
                print('Computing mean representations for evaluation...')
                loader_for_repr = representation_loader if representation_loader else self.traineval_loader
                # Representations are computed based on the training data's structure
                class_representations = self._compute_representations(loader_for_repr)

                if eval_on_train:
                    print('Evaluating on training set...')
                    train_acc = self._evaluate(class_representations, self.traineval_loader)
                    print(f'Epoch {epoch}: Train Top1 Acc: {train_acc * 100:.2f}%')

                if eval_on_test:
                    print('Evaluating on test set...')
                    test_acc = self._evaluate(class_representations, self.test_loader)
                    print(f'Epoch {epoch}: Test Accuracy: {test_acc * 100:.2f}%')

            end_time = time.time()
            print(f'Epoch time: {end_time - start_time:.1f} seconds')
        
        model_name = self.model.__class__.__name__
        removed_class_suffix = f'_removed_class_{REMOVE_CLASS}' if REMOVE_CLASS else ''
        torch.save(self.model.state_dict(), f'weights/{model_name}_{test_acc}_{removed_class_suffix}.pth')

    def evaluate(self, weights_path):
        """Evaluate the model with the given weights."""
        print(f'Evaluating {self.model.__class__.__name__} with weights from {weights_path}')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)

        class_representations = self._compute_representations(self.traineval_loader)
        test_acc = self._evaluate(class_representations, self.test_loader)
        print(f'Test Accuracy: {test_acc * 100:.2f}%')

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
            

if __name__ == '__main__':
    device = Trainer.get_device()

    if REMOVE_CLASS:
        remove_class = 0  # Example: remove class 0
        print(f'Removing class {remove_class} from the dataset.')
        trainset = MNISTMetricDataset(root=MNIST_DOWNLOAD_ROOT, split='train', remove_class=remove_class)
    else:
        trainset = MNISTMetricDataset(root=MNIST_DOWNLOAD_ROOT, split='train')

    testset = MNISTMetricDataset(root=MNIST_DOWNLOAD_ROOT, split='test')
    valset = MNISTMetricDataset(root=MNIST_DOWNLOAD_ROOT, split='traineval')

    print(f'Fitting PCA directly from images...')
    test_repr = testset.images.view(-1, 28 * 28)
    visualize_embeddings(test_repr, testset.targets)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SimpleMetricEmbedding(input_channels=1, emb_size=EMB_SIZE)
    # model = IdentityModel()  # Using IdentityModel for no trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(model, optimizer, train_loader, test_loader, val_loader, device, NUM_CLASSES, EMB_SIZE)
    # # trainer.train(epochs=EPOCHS, eval_on_test=EVAL_ON_TEST, eval_on_train=EVAL_ON_TRAIN)
    trainer.evaluate(weights_path='weights/SimpleMetricEmbedding_0.9714_.pth')
    
    with torch.no_grad():
        test_rep = model.get_features(testset.images.to(device))
        visualize_embeddings(test_rep, testset.targets)
