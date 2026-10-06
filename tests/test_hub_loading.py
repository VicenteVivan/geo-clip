import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from geoclip import GeoCLIP, LocationEncoder
from geoclip.model import _hub


@pytest.fixture
def fake_backbone(monkeypatch, tmp_path):
    calls = []

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def get_image_features(self, pixel_values):
            return pixel_values

    def load_model(path):
        calls.append(("model", path))
        return Backbone()

    def load_processor(path):
        calls.append(("processor", path))
        return SimpleNamespace()

    def download(**kwargs):
        calls.append(("download", kwargs))
        return str(tmp_path / "clip")

    module = importlib.import_module("geoclip.model.image_encoder")
    monkeypatch.setattr(module.CLIPModel, "from_pretrained", load_model)
    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(_hub, "snapshot_download", download)
    return calls


@pytest.fixture(scope="module")
def location_release(tmp_path_factory):
    folder = tmp_path_factory.mktemp("location-release")
    # Two scales ensure the loader actually honors config rather than defaults.
    model = LocationEncoder(sigma=[2, 8], from_pretrained=False)
    save_file(model.state_dict(), folder / "model.safetensors")
    config = dict(
        format_version=1,
        sigma=[2, 8],
        model_type="geoclip-location-encoder",
    )
    (folder / "config.json").write_text(json.dumps(config))
    return folder, model


@pytest.fixture
def main_release(tmp_path, location_release, fake_backbone):
    _, location = location_release
    model = GeoCLIP(from_pretrained=False)
    model.location_encoder = location
    model.gps_gallery = torch.tensor([[40.7128, -74.006], [34.0522, -118.2437]])
    weights = {"logit_scale": model.logit_scale.detach()}
    for prefix, module in (
        ("location_encoder.", location),
        ("image_encoder.mlp.", model.image_encoder.mlp),
    ):
        weights.update(
            {prefix + key: value for key, value in module.state_dict().items()}
        )
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "gallery.csv").write_text(
        "LAT,LON\n40.7128,-74.0060\n34.0522,-118.2437\n"
    )
    config = json.loads((location_release[0] / "config.json").read_text())
    config.update(
        model_type="geoclip-geolocation",
        gallery_file="gallery.csv",
        backbone=dict(repo_id="openai/clip-vit-large-patch14", revision="a" * 40),
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    fake_backbone.clear()
    return tmp_path, model


def test_location_local_parity_and_no_backbone(location_release, monkeypatch):
    folder, original = location_release
    module = importlib.import_module("geoclip.model.image_encoder")
    monkeypatch.setattr(
        module.CLIPModel, "from_pretrained", lambda *a, **k: pytest.fail("Loaded CLIP")
    )
    monkeypatch.setattr(
        _hub, "snapshot_download", lambda *a, **k: pytest.fail("Used network")
    )
    loaded = LocationEncoder.from_pretrained(folder, local_files_only=True)
    assert not loaded.training and loaded.sigma == [2, 8]
    assert all(
        not param.requires_grad
        for key, param in loaded.named_parameters()
        if key.endswith(".b")
    )
    points = torch.tensor(
        [[40.7128, -74.006], [34.0522, -118.2437], [90.0, 180.0], [-90.0, -180.0]]
    )
    with torch.no_grad():
        torch.testing.assert_close(original(points), loaded(points), rtol=0, atol=0)
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    loaded.train()
    loaded(points).sum().backward()
    assert loaded.LocEnc0.head[0].weight.grad is not None


def test_hub_download_arguments_and_private_auth(
    location_release, monkeypatch, tmp_path
):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(location_release[0])

    monkeypatch.setattr(_hub, "snapshot_download", download)
    LocationEncoder.from_pretrained(
        "owner/location",
        revision="v1",
        token=True,
        cache_dir=tmp_path,
        local_files_only=True,
    )
    assert calls == [
        dict(
            repo_id="owner/location",
            revision="v1",
            token=True,
            cache_dir=tmp_path,
            local_files_only=True,
            allow_patterns=["config.json", "model.safetensors", "*.csv"],
        )
    ]


def test_main_parity_pinned_backbone_and_gallery(main_release, fake_backbone, tmp_path):
    folder, original = main_release
    loaded = GeoCLIP.from_pretrained(
        folder, cache_dir=tmp_path, token=False, local_files_only=True, device="cpu"
    )
    assert not loaded.training and loaded.gps_gallery.shape == (2, 2)
    assert torch.equal(loaded.gps_gallery, original.gps_gallery)
    assert fake_backbone == [
        (
            "download",
            dict(
                repo_id="openai/clip-vit-large-patch14",
                revision="a" * 40,
                token=False,
                cache_dir=tmp_path,
                local_files_only=True,
                allow_patterns=["*.json", "merges.txt", "model.safetensors"],
            ),
        ),
        ("model", str(tmp_path / "clip")),
        ("processor", str(tmp_path / "clip")),
    ]
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    images = torch.randn(2, 768)
    with torch.no_grad():
        torch.testing.assert_close(
            original(images, original.gps_gallery),
            loaded(images, loaded.gps_gallery),
            rtol=0,
            atol=0,
        )
    assert loaded.gps_queue.shape == (2, 4096)
    assert not any(
        param.requires_grad for param in loaded.image_encoder.CLIP.parameters()
    )
    loaded.train()
    loaded(images, loaded.gps_gallery).sum().backward()
    assert loaded.logit_scale.grad is not None
    assert loaded.image_encoder.mlp[0].weight.grad is not None


def test_existing_constructors_still_load_bundled_weights(fake_backbone):
    model = GeoCLIP()
    location = LocationEncoder()
    assert model.training and location.training
    assert model.gps_gallery.shape == (100000, 2)
    for key, value in location.state_dict().items():
        assert torch.equal(model.location_encoder.state_dict()[key], value)
    assert len(fake_backbone) == 2
    untrained = GeoCLIP(from_pretrained=False, queue_size=3)
    assert untrained.gps_queue.shape == (2, 3)


@pytest.mark.parametrize(
    "field,value",
    [("format_version", 2), ("model_type", "other")],
)
def test_invalid_location_release_rejected(location_release, tmp_path, field, value):
    config = json.loads((location_release[0] / "config.json").read_text())
    config[field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        LocationEncoder.from_pretrained(tmp_path)


@pytest.mark.parametrize("change", ["missing", "shape"])
def test_invalid_fourier_tensor_rejected(location_release, tmp_path, change):
    config = json.loads((location_release[0] / "config.json").read_text())
    weights = load_file(location_release[0] / "model.safetensors")
    key = "LocEnc0.capsule.0.b"
    if change == "missing":
        del weights[key]
    else:
        weights[key] = weights[key][:1]
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError):
        LocationEncoder.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "gallery_path",
        "missing_tensor",
        "unexpected_tensor",
        "scalar_shape",
        "mlp_shape",
    ],
)
def test_main_release_rejects_inconsistency(main_release, fake_backbone, change):
    folder, _ = main_release
    config = json.loads((folder / "config.json").read_text())
    if change == "gallery_path":
        config["gallery_file"] = "../gallery.csv"
    else:
        weights = load_file(folder / "model.safetensors")
        if change == "missing_tensor":
            del weights["logit_scale"]
        elif change == "unexpected_tensor":
            weights["unexpected"] = torch.ones(1)
        elif change == "scalar_shape":
            weights["logit_scale"] = torch.ones(1)
        else:
            weights["image_encoder.mlp.0.weight"] = torch.ones(1, 1)
        save_file(weights, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps(config))
    with pytest.raises((ValueError, RuntimeError)):
        GeoCLIP.from_pretrained(folder)
    if change == "gallery_path":
        assert not fake_backbone


def test_missing_local_directory_never_uses_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _hub, "snapshot_download", lambda *a, **k: pytest.fail("Used network")
    )
    with pytest.raises(FileNotFoundError, match="Local model directory"):
        LocationEncoder.from_pretrained(tmp_path / "missing")


def test_missing_weights_reported(location_release, tmp_path):
    (tmp_path / "config.json").write_text(
        (location_release[0] / "config.json").read_text()
    )
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        LocationEncoder.from_pretrained(tmp_path)
