import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv"))
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        opt = self.opt
        gps_batch_size = gps.shape[0]
        batch_size = opt.batch_size

        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % batch_size == 0

        # Replace the GPS from ptr to ptr+batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def append_gps_queue_features(self, gps_features):
        """ Compute the GPS queue features and append them to the given GPS features."""

        # Get the GPS queue features
        location_queue = self.gps_queue.t().detach()
        gps_queue_features = self.location_encoder(location_queue)
        gps_queue_features = F.normalize(gps_queue_features, dim=1)

        # Concatenate Features (GPS Features & GPS Queue Features)
        gps_features = torch.cat([gps_features, gps_queue_features], dim=0)

        # Update GPS queue
        self._dequeue_and_enqueue(gps)

        return gps_features
                                             
    def forward(self, image, location):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)
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
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image, logits_per_location = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k prediction
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob