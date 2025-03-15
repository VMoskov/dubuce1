import cv2
import numpy as np
import torch


def draw_conv_filters(epoch, step, conv_layer, save_dir):
    weights = conv_layer.weight.data.cpu().numpy()
    num_filters = weights.shape[0]
    num_channels = weights.shape[1]
    k = weights.shape[2]

    assert weights.shape[2] == weights.shape[3]  # Ensure square filters
    
    num_rows, num_cols = 2, 8  
    assert num_filters >= num_rows * num_cols, "Not enough filters to fill the grid"

    border_size = 1  
    filter_images = []
    
    for i in range(num_rows * num_cols):
        img = weights[i, 0, :, :]  
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype(np.uint8)
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  
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


def train(model, criterion, optimizer, trainloader, valloader, config, SAVE_DIR):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    lr_policy = config['lr_policy']

    losses = []
    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0
        model.train()

        if epoch in lr_policy:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_policy[epoch]['lr']

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                draw_conv_filters(epoch, i*batch_size, model.layers[0], SAVE_DIR)

        accuracy = evaluate(model, valloader)
        print(f'Epoch {epoch}, validation accuracy: {accuracy}, loss: {epoch_loss}')
        losses.append(epoch_loss)

    return model, losses


def evaluate(model, dataloader):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy