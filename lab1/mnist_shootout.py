import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from pt_deep import PTDeep
import matplotlib.pyplot as plt
import data
import sklearn.svm as svm


param_epochs = 10
param_batch_size = 64

device = 'mps'


dataset_root = '/tmp/mnist'
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().item() + 1

# validation split 1/5 of the training set
N_val = N // 5
indices = torch.randperm(N)
x_val, y_val = x_train[indices[:N_val]], y_train[indices[:N_val]]

x_train, y_train = x_train[indices[N_val:]], y_train[indices[N_val:]]
N = x_train.shape[0]


def train_mb(model, x, y, x_val, y_val, epochs, batch_size):
    model.train()
    n_samples = x.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)

    losses = []
    best_model = None
    best_accuracy = 0

    for epoch in range(epochs):
        # shuffle the data
        indices = torch.randperm(n_samples)
        x, y = x[indices], y[indices]
        epoch_loss = 0

        for i in range(0, n_samples, batch_size):  # batch training
            x_batch = x[i:i+batch_size].view(-1, D)
            y_batch = y[i:i+batch_size]
            y_batch = nn.functional.one_hot(y_batch, num_classes=C).float()

            optimizer.zero_grad()
            loss = model.get_loss(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)

        # evaluate the model
        with torch.no_grad():
            model.eval()
            accuracy, precision, recall, val_loss = eval_mb(model, x_val, y_val)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model.state_dict()
            print(f'epoch: {epoch}, loss: {epoch_loss:.5f}, accuracy: {accuracy:.5f}, precision: {precision:.5f}, recall: {recall:.5f}')
        
        scheduler.step()
            
    model.load_state_dict(best_model)
    return model, losses


def eval_mb(model, x, y):
    x_val = x.view(-1, D)
    y_val = torch.tensor(y, dtype=torch.long)
    y_val = nn.functional.one_hot(y_val, num_classes=C).float()

    logits = model(x_val)
    y_pred = logits.argmax(dim=1)  # argmax over logits is the same as argmax over probs
    loss = model.get_loss(x_val, y_val)
    
    acc, pr, M = data.eval_perf_multi(y, y_pred.numpy())
    recall, precision = zip(*pr)
    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(recall)
    return acc, avg_precision, avg_recall, loss.item()


def plot_images(images, titles, plot_title):
    plt.suptitle(plot_title)
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].numpy(), cmap='gray')
        plt.title(titles[i].item())
        plt.axis('off')
    plt.show()
    


if __name__ == '__main__':
    # show the first 10 images
    plot_images(x_train, y_train, 'MNIST dataset')

    configs = [[784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10]]

    # PTDeep
    for config in configs:
        model = PTDeep(config)
        print(f'training PTDeep model with config: {config}')
        model, losses = train_mb(model, x_train, y_train, x_val, y_val, epochs=param_epochs, batch_size=param_batch_size)
        plt.plot(losses)
        plt.title(f'Epoch loss for config: {config}')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        accuracy, precision, recall, val_loss = eval_mb(model, x_test, y_test)
        print('-'*100)
        print('Best model evaluation:')
        print(f'accuracy: {accuracy:.5f}, loss: {val_loss:.5f}, precision: {precision:.5f}, recall: {recall:.5f}')
        print('='*100)

        # visualise 10 images which bring the highest loss
        x_val_flat = x_val.view(-1, D)
        y_val_oh = nn.functional.one_hot(y_val, num_classes=C).float()
        probs = model(x_val_flat)
        y_pred = probs.argmax(dim=1)
        loss = model.per_sample_loss(x_val_flat, y_val_oh)
        loss, indices = torch.topk(loss, 10)
        plot_images(x_val[indices], y_pred[indices], 'Images with the highest loss')

    # random initialization (PTDeep)
    model = PTDeep([784, 100, 100, 10])
    print(f'Random initialization PTDeep model, config: [784, 100, 100, 10]')
    accuracy, precision, recall, val_loss = eval_mb(model, x_test, y_test)
    print(f'accuracy: {accuracy:.5f}, precision: {precision:.5f}, loss: {val_loss:.5f}')
    print('='*100)


    # SVM (default one-vs-one)
    x_train = x_train.view(-1, D).numpy()
    y_train = y_train.numpy()
    x_test = x_test.view(-1, D).numpy()
    y_test = y_test.numpy()

    model = svm.SVC()
    print('SVM model')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy, pr, M = data.eval_perf_multi(y_test, y_pred)
    recall, precision = zip(*pr)
    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(recall)
    print(f'accuracy: {accuracy:.5f}, precision: {avg_precision:.5f}, recall: {avg_recall:.5f}')
