import warnings

import torch
from torch import nn
from transformers import AutoProcessor, CLIPConfig, CLIPImageProcessor, CLIPModel

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
    def __init__(self, from_pretrained=True, *, clip_path=None):
        super().__init__()
        if from_pretrained:
            clip_model = "openai/clip-vit-large-patch14" if clip_path is None else clip_path
            self.CLIP = CLIPModel.from_pretrained(clip_model)
            self.image_processor = AutoProcessor.from_pretrained(clip_model)
        else:
            if clip_path is not None:
                raise ValueError("clip_path requires from_pretrained=True.")
            config = CLIPConfig(
                projection_dim=768,
                text_config=dict(
                    hidden_size=768,
                    intermediate_size=3072,
                    num_attention_heads=12,
                    projection_dim=768,
                    bos_token_id=0,
                    eos_token_id=2,
                ),
                vision_config=dict(
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    patch_size=14,
                    projection_dim=768,
                ),
            )
            self.CLIP = CLIPModel(config)
            self.image_processor = CLIPImageProcessor()
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        if from_pretrained:
            for param in self.CLIP.parameters():
                param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = _extract_image_features(self.CLIP.get_image_features(pixel_values=x))
        x = self.mlp(x)
        return x
