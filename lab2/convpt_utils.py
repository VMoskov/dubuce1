import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


def draw_image(ax, img, label, preds, loss, mean, std):
    #  show image and cancel out normalization
    mean = np.array(mean)
    std = np.array(std)

    ax.axis('off')
    ax.set_title(f'Loss: {loss:.2f}')
    img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    img = img * std + mean
    img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
    img = img * 255
    img = img.astype(np.uint8)
    ax.set_xlabel(f'Label: {label}, Top3 preds: {preds}')
    pred_str = '\n'.join(preds)
    ax.text(0.5, -0.1, f'Label: {label}\nTop3:\n{pred_str}', 
            fontsize=8, ha='center', va='top', transform=ax.transAxes)
    ax.imshow(img)


def show_highest_loss_images(model, dataloader, label_names, mean, std, num_images=20, criterion=nn.CrossEntropyLoss(reduction='none')):
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    losses = []
    images = []
    labels = []
    preds = []
    label_names = np.array(label_names)
    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, label) 

            _, top3 = torch.topk(outputs, 3, dim=1)

            losses.extend(loss.cpu().numpy())
            images.extend(inputs.cpu())
            # extract label names
            labels.extend(label_names[label.cpu().numpy()])
            preds.extend(label_names[top3.cpu().numpy()])

    losses = np.array(losses)
    indices = np.argsort(losses)[-num_images:]

    # 2 rows, 10 columns
    num_rows = 2
    num_cols = 10

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(num_rows):
        for j in range(num_cols):
            idx = indices[i * num_cols + j]
            ax = axes[i, j]
            img = images[idx]
            label = labels[idx]
            pred = preds[idx]
            loss = losses[idx]
            draw_image(ax, img, label, pred, loss, mean, std)
    plt.suptitle('Highest loss images')
    plt.show()


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def draw_conv_filters(epoch, step, conv_layer, save_dir):
    weights = conv_layer.weight.data.cpu().numpy()
    num_filters, num_channels, k, _ = weights.shape  # (filters, channels, height, width)
    
    assert k == weights.shape[3], "Filters should be square"
    
    num_rows, num_cols = 2, 8  
    assert num_filters >= num_rows * num_cols, "Not enough filters to fill the grid"

    border_size = 1  
    filter_images = []

    for i in range(num_rows * num_cols):
        img = weights[i]  # Shape: (C, k, k)

        # Normalize per-channel
        for c in range(num_channels):
            img[c] -= img[c].min()
            img[c] /= img[c].max() if img[c].max() > 0 else 1  # Avoid division by zero
            img[c] *= 255

        img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        img = img.astype(np.uint8)
        # img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # If grayscale (1 channel), convert to RGB by repeating channels
        if num_channels == 1:
            img = np.repeat(img, 3, axis=2)

        filter_images.append(img)

    filter_h, filter_w, _ = filter_images[0].shape

    # Create a blank black canvas with separators
    canvas_h = num_rows * filter_h + (num_rows - 1) * border_size
    canvas_w = num_cols * filter_w + (num_cols - 1) * border_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * (filter_w + border_size)
            y = row * (filter_h + border_size)
            canvas[y:y + filter_h, x:x + filter_w] = filter_images[row * num_cols + col]

    cv2.imwrite(f'{save_dir}/conv1_epoch_{epoch:02d}_step_{step:06d}_input_000.png', canvas)


def multiclass_hinge_loss(logits, target, delta=1.0):
    """
    Args:
        logits: torch.Tensor with shape (B, C), where B is batch size, and C is number of classes.
        target: torch.LongTensor with shape (B, ) representing ground truth labels.
        delta: Hyperparameter.
    Returns:
        Loss as scalar torch.Tensor.
    """
    batch_size = logits.size(0)
    num_classes = logits.size(1)

    # Create a mask for the target class
    target_mask = torch.zeros_like(logits)
    target_mask[torch.arange(batch_size), target] = 1  # Set the target class to 1

    correct_class_logits = torch.sum(logits * target_mask, dim=1, keepdim=True)

    margins = logits - correct_class_logits + delta
    margins *= (1 - target_mask)  # zero out the target class margin
    hinge_loss = torch.clamp(margins, min=0)  # Apply ReLU to the margins

    loss = torch.sum(hinge_loss, dim=1)  # Sum over classes
    loss = torch.mean(loss)  # Average over batch
    return loss


def train(model, criterion, optimizer, scheduler, trainloader, valloader, config, SAVE_DIR):
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    max_epochs = config['max_epochs']
    batch_size = config['batch_size']

    losses = []
    best_accuracy = 0
    best_model = None
    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0
        model.train()

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # clip avoid exploding gradients
            optimizer.step()

            if i % 100 == 0:
                ...#draw_conv_filters(epoch, i*batch_size, model.conv1, SAVE_DIR)


        train_accuracy, train_loss = evaluate(model, trainloader, criterion)
        val_accuracy, val_loss = evaluate(model, valloader, criterion)

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_accuracy]
        plot_data['valid_acc'] += [val_accuracy]
        plot_data['lr'] += [scheduler.get_last_lr()]

        accuracy = val_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()

        print(f'Epoch {epoch}, validation accuracy: {accuracy}, loss: {epoch_loss}')
        losses.append(epoch_loss)
        scheduler.step()

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
    plot_training_progress(SAVE_DIR, plot_data)
    return model, losses


def evaluate(model, dataloader, criterion):
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    eval_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    eval_loss /= len(dataloader.dataset)
    return accuracy, eval_loss