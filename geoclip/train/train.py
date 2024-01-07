import torch
from tqdm import tqdm

def train(train_dataloader, model, criterion, optimizer, scheduler, epoch, batch_size, device):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = torch.Tensor([i for i in range(batch_size).to(device)]).long()

    for i ,(imgs, gps) in bar:
        imgs = imgs.to(device)
        gps = gps.to(device)

        optimizer.zero_grad()

        # Forward pass
        img_features = model.image_encoder(imgs)
        gps_features = model.location_encoder(gps)
        
        # Normalize the features
        img_features = F.normalize(img_features, dim=1)
        gps_features = F.normalize(gps_features, dim=1)

        # Append Queue
        gps_features = model.append_gps_queue_features(gps_features)

        # Get the temperature
        temp = model.logit_scale.exp()

        # Compute the logits
        logits_img_gps = temp * (img_features @ gps_features.T)

        # Compute the loss
        loss = 0
        img_gps_loss = criterion(logits_img_gps, targets_img_gps)
        loss += img_gps_loss

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Update the progress bar
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    # Update the scheduler
    scheduler.step()
