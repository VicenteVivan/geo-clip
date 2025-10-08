import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import CLIPTokenizer

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

class GeoCLIPPlus(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096):
        """
        Initialize GeoCLIP-Plus with advanced multimodal geolocation capabilities.

        Args:
            from_pretrained (bool): Load pre-trained weights. Defaults to True.
            queue_size (int): Size of GPS gallery queue. Defaults to 4096.
        """
        super().__init__()
        
        # Core model components
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()
        
        # Text processing capability
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # GPS gallery and queue management
        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv"))
        
        # Initialize GPS queue for dynamic feature management
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Weights and device management
        self.weights_folder = os.path.join(file_dir, "weights")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained weights if specified
        if from_pretrained:
            self._load_weights()

    def to(self, device):
        """
        Move model and its components to specified device.

        Args:
            device (torch.device): Target compute device.

        Returns:
            GeoCLIPPlus: Model moved to specified device.
        """
        super().to(device)
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        # Ensure logit_scale and logit_bias are moved to the correct device
        self.logit_scale.data = self.logit_scale.data.to(device)
        # self.logit_bias.data = self.logit_bias.data.to(device) # Assuming logit_bias exists
        return self

    def _load_weights(self):
        """
        Load pre-trained weights for image encoder, location encoder, and logit scale.
        """
        try:
            self.image_encoder.mlp.load_state_dict(
                torch.load(os.path.join(self.weights_folder, "image_encoder_mlp_weights.pth"))
            )
            self.location_encoder.load_state_dict(
                torch.load(os.path.join(self.weights_folder, "location_encoder_weights.pth"))
            )
            self.logit_scale = nn.Parameter(
                torch.load(os.path.join(self.weights_folder, "logit_scale_weights.pth"))
            )
        except FileNotFoundError as e:
            print(f"Weight loading error: {e}")
            raise RuntimeError("Pre-trained weights could not be loaded.")

    def dequeue_and_enqueue(self, gps):
        """
        Update GPS queue dynamically during training.

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, (
            f"Queue size {self.queue_size} must be divisible by batch size {gps_batch_size}"
        )

        # Replace GPS in queue
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        """
        Retrieve current GPS queue.

        Returns:
            torch.Tensor: Transposed GPS queue
        """
        return self.gps_queue.t()

    def forward(self, image, location):
        """
        GeoCLIP-Plus forward pass for multimodal feature extraction.

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            torch.Tensor: Logits per image of shape (n, m)
        """
        # Feature extraction
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Compute cosine similarity with learnable temperature scaling
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    def get_image_embedding(self, image_path):
        """
        Generate a high-dimensional embedding for an input image.

        Args:
            image_path (str): Path to input image file.

        Returns:
            torch.Tensor: Normalized image embedding vector
        """
        # Open and preprocess image
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        # Extract and normalize image features
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = F.normalize(image_features, dim=1)

        return image_features.squeeze(0)

    def predict_from_text(self, text_query, top_k=5):
        """
        Predict geographic locations from a text description.

        Args:
            text_query (str): Descriptive text for location prediction.
            top_k (int, optional): Number of top predictions. Defaults to 5.

        Returns:
            tuple: Top predicted GPS coordinates and their probabilities
        """
        if not text_query:
            raise ValueError("Text query cannot be empty.")

        with torch.no_grad():
            # Tokenize and process text input
            text_inputs = self.tokenizer(
                text_query, 
                padding=True, 
                return_tensors="pt", 
                truncation=True
            ).to(self.device)

            # Extract text features
            text_features = self.image_encoder.CLIP.get_text_features(**text_inputs)
            text_features = self.image_encoder.mlp(text_features)
            text_features = F.normalize(text_features, dim=1)

            # Prepare GPS gallery
            gps_gallery = self.gps_gallery.to(self.device)
            location_features = self.location_encoder(gps_gallery)
            location_features = F.normalize(location_features, dim=1)

            # Compute similarity and probabilities
            logit_scale = self.logit_scale.exp()
            similarity = logit_scale * (text_features @ location_features.t())
            probs = similarity.softmax(dim=-1)

            # Extract top predictions
            top_pred = torch.topk(probs[0], top_k)
            top_pred_gps = gps_gallery[top_pred.indices]
            top_pred_prob = top_pred.values

        return top_pred_gps, top_pred_prob

    def predict(self, input_source, top_k=5):
        """
        Unified prediction method for images and text.

        Args:
            input_source (str): Image path or text query.
            top_k (int, optional): Number of top predictions. Defaults to 5.

        Returns:
            tuple: Predicted GPS coordinates and their probabilities
        """
        if isinstance(input_source, str):
            # Determine input type based on file extension
            _, ext = os.path.splitext(input_source)
            if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                return self._predict_from_image(input_source, top_k)
            else:
                # Assuming it's a text query if not a recognized image extension
                return self.predict_from_text(input_source, top_k)
        else:
            raise ValueError("Input must be an image path or text query.")

    def _predict_from_image(self, image_path, top_k=5):
        """
        Predict top GPS coordinates from an image.

        Args:
            image_path (str): Path to input image.
            top_k (int, optional): Number of top predictions. Defaults to 5.

        Returns:
            tuple: Top predicted GPS coordinates and their probabilities
        """
        # Preprocess image
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        # Prepare GPS gallery
        gps_gallery = self.gps_gallery.to(self.device)

        # Compute predictions
        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Extract top predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob
