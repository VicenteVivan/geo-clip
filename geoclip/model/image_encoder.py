import warnings

import torch
from torch import nn
from transformers import AutoProcessor, CLIPModel

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


def _extract_image_features(output):
    """Return projected CLIP features across Transformers 4.x and 5.x."""
    if isinstance(output, torch.Tensor):
        return output

    pooled_output = getattr(output, "pooler_output", None)
    if isinstance(pooled_output, torch.Tensor):
        return pooled_output

    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]

    raise TypeError(
        "CLIPModel.get_image_features returned an unsupported value of type "
        f"{type(output).__name__}."
    )


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = _extract_image_features(self.CLIP.get_image_features(pixel_values=x))
        x = self.mlp(x)
        return x
