import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))
        self.mlp.load_state_dict(torch.load("model/weights/image_encoder_mlp_weights.pth"))

    def preprocess_image(self, image):
        return self.image_processor(images=image, return_tensors="pt")

    def get_image_features(self, image):
        inputs = self.preprocess_image(image)
        return self.CLIP.get_image_features(**inputs)

    def forward(self, x):
        x = self.get_image_features(x)
        x = self.mlp(x)
        return x
