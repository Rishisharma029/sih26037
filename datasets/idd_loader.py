import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from .idd_schema import IDDImageSample, DETECTION_CLASS_TO_IDX

class IDDDataset(Dataset):
    def __init__(self, base_data_dir: str, split: str = "train", target_size: tuple = (256, 256)):
        self.base_data_dir = base_data_dir
        self.split = split
        self.target_size = target_size
        self.manifest_path = os.path.join(base_data_dir, f"{split}_manifest.json")

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.samples = [IDDImageSample.model_validate(item) for item in json.load(f)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img_full_path = os.path.join(self.base_data_dir, sample.file_path)
        mask_full_path = os.path.join(self.base_data_dir, sample.segmentation_mask_path) if sample.segmentation_mask_path else None

        img = Image.open(img_full_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize(self.target_size, Image.BILINEAR)
        img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32).permute(2, 0, 1) / 255.0

        if mask_full_path and os.path.exists(mask_full_path):
            mask_img = Image.open(mask_full_path)
            mask_resized = mask_img.resize(self.target_size, Image.NEAREST)
            mask_tensor = torch.tensor(np.array(mask_resized), dtype=torch.long)
        else:
            mask_tensor = torch.zeros(self.target_size[1], self.target_size[0], dtype=torch.long)

        scale_x = self.target_size[0] / orig_w
        scale_y = self.target_size[1] / orig_h

        boxes = []
        labels = []
        for box in sample.boxes:
            boxes.append([
                box.xmin * scale_x,
                box.ymin * scale_y,
                box.xmax * scale_x,
                box.ymax * scale_y
            ])
            labels.append(DETECTION_CLASS_TO_IDX[box.label.value])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": sample.image_id
        }

def get_idd_dataloader(base_data_dir: str, split: str = "train", batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    dataset = IDDDataset(base_data_dir=base_data_dir, split=split)
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])
        boxes = [item["boxes"] for item in batch]
        labels = [item["labels"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        return {
            "images": images,
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "image_ids": image_ids
        }
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
