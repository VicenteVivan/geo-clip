import torch
from torch import nn
from tqdm import tqdm


def train(
    train_dataloader,
    model,
    optimizer,
    epoch,
    batch_size,
    device,
    scheduler=None,
    criterion=None,
):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    criterion = criterion or nn.CrossEntropyLoss()

    for _, (imgs, gps) in bar:
        imgs = imgs.to(device)
        gps = gps.to(device)
        gps_queue = model.get_gps_queue()
        targets_img_gps = torch.arange(imgs.shape[0], device=device)

        optimizer.zero_grad()

        # Append GPS Queue & Queue Update
        gps_all = torch.cat([gps, gps_queue], dim=0)
        model.dequeue_and_enqueue(gps)

        # Forward pass
        logits_img_gps = model(imgs, gps_all)

        # Compute the loss
        img_gps_loss = criterion(logits_img_gps, targets_img_gps)
        loss = img_gps_loss

        # Backpropagate
        loss.backward()
        optimizer.step()

        bar.set_description(f"Epoch {epoch} loss: {loss.item():.5f}")

    if scheduler is not None:
        scheduler.step()
