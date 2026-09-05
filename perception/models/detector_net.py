import torch
import torch.nn as nn

class IDDObjectDetector(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        bboxes = torch.sigmoid(self.bbox_regressor(feat)) * 256.0
        return logits, bboxes
