import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import file_dir, load_gps_data


class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder(from_pretrained=False)

        gps_gallery = load_gps_data(
            os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv")
        )
        self.register_buffer("gps_gallery", gps_gallery, persistent=False)
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(
            torch.load(
                f"{self.weights_folder}/image_encoder_mlp_weights.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.location_encoder.load_state_dict(
            torch.load(
                f"{self.weights_folder}/location_encoder_weights.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.logit_scale = nn.Parameter(
            torch.load(
                f"{self.weights_folder}/logit_scale_weights.pth",
                map_location="cpu",
                weights_only=True,
            )
        )

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)

        if gps_batch_size >= self.queue_size:
            self.gps_queue.copy_(gps[-self.queue_size :].t())
            self.gps_queue_ptr.zero_()
            return

        end_ptr = gps_ptr + gps_batch_size
        if end_ptr <= self.queue_size:
            self.gps_queue[:, gps_ptr:end_ptr] = gps.t()
        else:
            first_chunk = self.queue_size - gps_ptr
            self.gps_queue[:, gps_ptr:] = gps[:first_chunk].t()
            self.gps_queue[:, : end_ptr - self.queue_size] = gps[first_chunk:].t()

        gps_ptr = end_ptr % self.queue_size
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()

    def forward(self, image, location):
        """GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        if not 1 <= top_k <= len(self.gps_gallery):
            raise ValueError(
                f"top_k must be between 1 and {len(self.gps_gallery)}, got {top_k}."
            )

        with Image.open(image_path) as image:
            image = self.image_encoder.preprocess_image(image.convert("RGB"))
        image = image.to(self.logit_scale.device)

        logits_per_image = self.forward(image, self.gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1)

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery.index_select(0, top_pred.indices[0]).cpu()
        top_pred_prob = top_pred.values[0].cpu()

        return top_pred_gps, top_pred_prob
