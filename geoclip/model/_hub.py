"""Load GeoCLIP weights and configuration from the Hugging Face Hub."""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import torch

from .misc import load_gps_data


def read_release(source, model_type, **download_options):
    """Read configuration and weights from the same snapshot."""
    path = Path(source).expanduser()
    if not path.is_dir():
        if (
            isinstance(source, os.PathLike)
            or path.is_absolute()
            or str(source).startswith(".")
        ):
            raise FileNotFoundError(f"Local model directory does not exist: {source}")
        path = Path(
            snapshot_download(
                repo_id=str(source),
                allow_patterns=["config.json", "model.safetensors", "*.csv"],
                **download_options,
            )
        )
    config = json.loads((path / "config.json").read_text())
    if (
        not isinstance(config, dict)
        or config.get("model_type") != model_type
        or type(config.get("format_version")) is not int
        or config["format_version"] != 1
    ):
        raise ValueError(f"Expected a {model_type} release with format_version=1.")
    return path, config, load_file(str(path / "model.safetensors"), device="cpu")


def read_gallery(path, config):
    filename = config.get("gallery_file")
    if (
        not isinstance(filename, str)
        or not filename.endswith(".csv")
        or Path(filename).name != filename
    ):
        raise ValueError(
            "config.json gallery_file must name a CSV file inside the release."
        )
    return load_gps_data(path / filename)


def download_backbone(config, **download_options):
    backbone = config["backbone"]
    return snapshot_download(
        repo_id=backbone["repo_id"],
        revision=backbone["revision"],
        allow_patterns=["*.json", "merges.txt", "model.safetensors"],
        **download_options,
    )


def load_geoclip_weights(model, weights):
    expected = {"logit_scale"}
    expected.update(
        "location_encoder." + key for key in model.location_encoder.state_dict()
    )
    expected.update(
        "image_encoder.mlp." + key for key in model.image_encoder.mlp.state_dict()
    )
    if set(weights) != expected:
        missing, unexpected = sorted(expected - weights.keys()), sorted(
            weights.keys() - expected
        )
        raise ValueError(
            f"GeoCLIP checkpoint keys do not match: missing={missing}, unexpected={unexpected}."
        )
    for prefix, module in (
        ("location_encoder.", model.location_encoder),
        ("image_encoder.mlp.", model.image_encoder.mlp),
    ):
        module.load_state_dict(
            {
                key[len(prefix) :]: value
                for key, value in weights.items()
                if key.startswith(prefix)
            },
            strict=True,
        )
    if weights["logit_scale"].shape != model.logit_scale.shape:
        raise ValueError("logit_scale must be a scalar tensor.")
    with torch.no_grad():
        model.logit_scale.copy_(weights["logit_scale"])
