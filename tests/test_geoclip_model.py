import os
import torch
import pytest
from PIL import Image

# Add the project root to PYTHONPATH to allow importing geoclip
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geoclip.model import GeoCLIPPlus
from geoclip.model.misc import file_dir # To get base path for images

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def model():
    # Ensure weights are downloaded if this is the first run or they are missing.
    # This might take time if weights need to be fetched by the model's __init__
    # For now, assume weights are present as per the original repo structure.
    try:
        model_instance = GeoCLIPPlus(from_pretrained=True)
        model_instance.to(DEVICE)
        model_instance.eval() # Set to eval mode for testing predictions
        return model_instance
    except Exception as e:
        pytest.fail(f"Failed to initialize GeoCLIPPlus model: {e}")

def test_model_initialization_and_device(model):
    assert model is not None
    assert isinstance(model.logit_scale, torch.Tensor)
    # Check if a parameter (e.g., logit_scale) is on the correct device
    # For parameters, device is directly accessible. For buffers/tensors, it's .device
    assert model.logit_scale.device == DEVICE
    # Check a submodule
    assert next(model.image_encoder.parameters()).device == DEVICE
    assert next(model.location_encoder.parameters()).device == DEVICE


def test_predict_from_image(model):
    # Using the provided sample image
    # Construct the full path to the image
    # The 'geoclip' module is in the parent directory of 'file_dir' if file_dir points to geoclip/model/
    # So, project_root should be parent of parent of file_dir
    project_root = os.path.dirname(os.path.dirname(file_dir))
    image_path = os.path.join(project_root, "geoclip", "images", "Kauai.png")

    assert os.path.exists(image_path), f"Test image not found at {image_path}"

    top_k = 3
    try:
        pred_gps, pred_prob = model.predict(image_path, top_k=top_k)
    except Exception as e:
        pytest.fail(f"model.predict() with image failed: {e}")

    assert pred_gps is not None
    assert pred_prob is not None
    assert isinstance(pred_gps, torch.Tensor)
    assert isinstance(pred_prob, torch.Tensor)
    assert pred_gps.shape == (top_k, 2) # top_k predictions, 2 for lat/lon
    assert pred_prob.shape == (top_k,)

def test_predict_from_text(model):
    text_query = "a beach with palm trees"
    top_k = 3
    try:
        pred_gps, pred_prob = model.predict(text_query, top_k=top_k) # Using the unified predict method
    except Exception as e:
        pytest.fail(f"model.predict() with text failed: {e}")

    assert pred_gps is not None
    assert pred_prob is not None
    assert isinstance(pred_gps, torch.Tensor)
    assert isinstance(pred_prob, torch.Tensor)
    assert pred_gps.shape == (top_k, 2)
    assert pred_prob.shape == (top_k,)

def test_predict_invalid_input(model):
    with pytest.raises(ValueError):
        model.predict(12345) # Invalid input type

    with pytest.raises(ValueError): # Assuming predict_from_text raises ValueError for empty string
        model.predict_from_text("", top_k=1)


# To run these tests, navigate to the root of the project in your terminal
# and execute the command: pytest
