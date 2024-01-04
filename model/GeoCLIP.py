import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.location_encoder import LocationEncoder
from model.image_encoder import ImageEncoder
from model.misc import load_gps_data

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = load_gps_data("model/gps_gallery/coordinates_100K.csv")

        if from_pretrained:
            self.load_weights()
            
    def load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load("model/weights/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load("model/weights/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load("model/weights/logit_scale_weights.pth"))
                                             
    def forward(self, image, location):
        """ Forward pass of GeoCLIP

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
        
        # Cosine similarity (Image Features - Location Feature Queue)
        logits_per_image = logit_scale * (image_features @ location_features.t())
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location

    def predict(self, image, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image (PIL.Image): Input image
            top_k (int): Number of top predictions to return
        """
        image = self.image_encoder.preprocess_image(image)

        logits_per_image, logits_per_location = self.forward(image, self.gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1)

        # Get top k prediction
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob