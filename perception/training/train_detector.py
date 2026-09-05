import torch
import torch.nn as nn
import torch.optim as optim
from datasets.idd_loader import get_idd_dataloader
from perception.models.detector_net import IDDObjectDetector

def train_idd_detector(base_data_dir: str, epochs: int = 1, batch_size: int = 4, lr: float = 1e-3) -> dict:
    train_loader = get_idd_dataloader(base_data_dir, split="train", batch_size=batch_size, shuffle=True)
    val_loader = get_idd_dataloader(base_data_dir, split="val", batch_size=batch_size, shuffle=False)

    model = IDDObjectDetector(num_classes=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cls_criterion = nn.BCEWithLogitsLoss()
    bbox_criterion = nn.SmoothL1Loss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            images = batch["images"]
            batch_sz = images.size(0)

            target_cls = torch.zeros((batch_sz, 8))
            target_bbox = torch.zeros((batch_sz, 4))

            for i in range(batch_sz):
                lbls = batch["labels"][i]
                bxs = batch["boxes"][i]
                if len(lbls) > 0:
                    for l in lbls:
                        target_cls[i, l] = 1.0
                    target_bbox[i] = bxs[0]

            optimizer.zero_grad()
            logits, bboxes = model(images)
            loss_cls = cls_criterion(logits, target_cls)
            loss_box = bbox_criterion(bboxes, target_bbox) * 0.01
            loss = loss_cls + loss_box

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_train_loss = total_loss / max(1, batches)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"]
                batch_sz = images.size(0)
                target_cls = torch.zeros((batch_sz, 8))
                for i in range(batch_sz):
                    for l in batch["labels"][i]:
                        target_cls[i, l] = 1.0
                logits, _ = model(images)
                val_loss += cls_criterion(logits, target_cls).item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        history["val_loss"].append(avg_val_loss)

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "model": model
    }
