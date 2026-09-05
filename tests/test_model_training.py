import pytest
import torch
from datasets.sample_generator import generate_mock_idd_dataset
from perception.models.detector_net import IDDObjectDetector
from perception.models.segmenter_net import IDDSegmentationModel
from perception.training.train_detector import train_idd_detector
from perception.training.train_segmenter import train_idd_segmenter

@pytest.fixture(scope="session")
def model_test_data(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("idd_model_data"))
    generate_mock_idd_dataset(data_dir, num_train=8, num_val=2, num_test=2)
    return data_dir

def test_detector_forward_pass():
    model = IDDObjectDetector(num_classes=8)
    dummy_input = torch.randn(2, 3, 256, 256)
    logits, bboxes = model(dummy_input)
    assert logits.shape == (2, 8)
    assert bboxes.shape == (2, 4)

def test_segmenter_forward_pass():
    model = IDDSegmentationModel(num_classes=6)
    dummy_input = torch.randn(2, 3, 256, 256)
    mask_logits = model(dummy_input)
    assert mask_logits.shape == (2, 6, 256, 256)

def test_detector_training_loop(model_test_data):
    res = train_idd_detector(base_data_dir=model_test_data, epochs=1, batch_size=2)
    assert "final_train_loss" in res
    assert res["final_train_loss"] > 0.0

def test_segmenter_training_loop(model_test_data):
    res = train_idd_segmenter(base_data_dir=model_test_data, epochs=1, batch_size=2)
    assert "final_train_loss" in res
    assert res["final_train_loss"] > 0.0
