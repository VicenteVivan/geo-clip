import numpy as np
import torch
from geopy.distance import geodesic
from tqdm import tqdm


def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    if total == 0:
        raise ValueError("targets must not be empty")

    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = geodesic(gps_gallery[preds[i]], targets[i]).km
        gd_avg += gd
        if gd <= dis:
            correct += 1

    gd_avg /= total
    return correct / total, gd_avg


def eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery.to(device)

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)

            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()

            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1]  # km
    accuracy_results = {}
    gps_gallery = gps_gallery.detach().cpu().numpy()
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(
            f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}"
        )
        accuracy_results[f"acc_{dis}_km"] = acc

    return accuracy_results
