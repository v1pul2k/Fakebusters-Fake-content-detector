# predict_image.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = "cnn_resnet18_best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = IMAGENET_MEAN
        self.std  = IMAGENET_STD

        self.base = models.resnet18(pretrained=True)
        in_feat = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def forward(self, x):
        return self.base(x)


def load_model_and_threshold():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = CNNModel().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    threshold = ckpt.get('threshold', 0.5)
    idx_to_class = {v: k for k, v in ckpt['class_to_idx'].items()}
    return model, threshold, idx_to_class


@torch.no_grad()
def predict_image(img_path, model, threshold):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x = model.transform(img).unsqueeze(0).to(DEVICE)

    logit = model(x)
    prob = torch.sigmoid(logit).item()

    pred_idx = 1 if prob > threshold else 0
    return prob, pred_idx


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py path/to/image.jpg")
        exit()

    img_path = sys.argv[1]

    model, threshold, idx_to_class = load_model_and_threshold()
    prob, pred_idx = predict_image(img_path, model, threshold)

    print(json.dumps({
        "image": img_path,
        "prob_real": prob,
        "threshold": threshold,
        "prediction": idx_to_class[pred_idx]
    }, indent=2))
