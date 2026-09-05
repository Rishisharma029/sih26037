import torch
import torch.nn as nn
import torch.optim as optim
from datasets.idd_loader import get_idd_dataloader
from perception.models.segmenter_net import IDDSegmentationModel

def train_idd_segmenter(base_data_dir: str, epochs: int = 1, batch_size: int = 4, lr: float = 1e-3) -> dict:
    train_loader = get_idd_dataloader(base_data_dir, split="train", batch_size=batch_size, shuffle=True)
    val_loader = get_idd_dataloader(base_data_dir, split="val", batch_size=batch_size, shuffle=False)

    model = IDDSegmentationModel(num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            images = batch["images"]
            masks = batch["masks"]

            optimizer.zero_grad()
            pred_masks = model(images)
            loss = criterion(pred_masks, masks)

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
                masks = batch["masks"]
                pred_masks = model(images)
                val_loss += criterion(pred_masks, masks).item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        history["val_loss"].append(avg_val_loss)

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "model": model
    }
