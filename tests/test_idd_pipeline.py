import pytest
import torch
from datasets.sample_generator import generate_mock_idd_dataset
from datasets.idd_loader import IDDDataset, get_idd_dataloader
from datasets.analyzer import IDDDatasetAnalyzer

@pytest.fixture(scope="session")
def setup_mock_data(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("idd_test_data"))
    generate_mock_idd_dataset(data_dir, num_train=12, num_val=4, num_test=4)
    return data_dir

def test_idd_dataset_loading(setup_mock_data):
    dataset = IDDDataset(base_data_dir=setup_mock_data, split="train", target_size=(128, 128))
    assert len(dataset) == 12
    sample = dataset[0]
    assert sample["image"].shape == (3, 128, 128)
    assert sample["mask"].shape == (128, 128)
    assert isinstance(sample["boxes"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)

def test_idd_dataloader_batching(setup_mock_data):
    loader = get_idd_dataloader(base_data_dir=setup_mock_data, split="val", batch_size=2)
    batch = next(iter(loader))
    assert batch["images"].shape == (2, 3, 256, 256)
    assert batch["masks"].shape == (2, 256, 256)

def test_dataset_analysis(setup_mock_data):
    analyzer = IDDDatasetAnalyzer(setup_mock_data)
    stats = analyzer.analyze()
    assert stats["total_images"] == 20
    assert "train" in stats["splits"]
    assert "val" in stats["splits"]
    assert "test" in stats["splits"]
