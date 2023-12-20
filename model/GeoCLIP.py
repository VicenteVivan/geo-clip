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
    def __init__(self,  input_resolution=224, opt=None, dim=512):
        super().__init__()

        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = load_gps_data("gps_gallery_100K.csv")

        # Load weights
        self.logit_scale = nn.Parameter(torch.load("model/weights/logit_scale_weights.pth"))
        self.location_encoder.load_state_dict(torch.load("model/weights/location_encoder_weights.pth"))
                                             
    def forward(self, image, location):
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
        logits_per_image, logits_per_location = self.forward(image, self.gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1)

        # Get top k prediction
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob

if __name__ == "__main__":
    image = Image.open("image.png")
    model = GeoCLIP()

    model.eval()
    with torch.no_grad():
        top_pred_gps, top_pred_prob = model.predict(image)
        print(f"Predicted GPS: {top_pred_gps}")
        print(f"Prediction Probability: {top_pred_prob}")