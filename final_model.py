# train_eval.py
import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------- CONFIG ----------------
CONFIG = {
    'base_path': 'real_vs_fake/real-vs-fake',
    'batch_size': 32,
    'num_epochs': 15,
    'learning_rate': 3e-4,
    'num_workers': 4,
    'use_mixed_precision': True,
    'early_stopping_patience': 5,
    'weight_decay': 0.01,
    'threshold_metric': 'youden',  # 'youden' | 'f1'
    'model_name': 'cnn_resnet18',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- DATA ----------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}

def load_datasets(base_path):
    train_ds = datasets.ImageFolder(os.path.join(base_path, 'train'), transform=data_transforms['train'])
    valid_ds = datasets.ImageFolder(os.path.join(base_path, 'valid'), transform=data_transforms['valid'])
    test_ds  = datasets.ImageFolder(os.path.join(base_path, 'test'),  transform=data_transforms['test'])
    return train_ds, valid_ds, test_ds

train_ds, valid_ds, test_ds = load_datasets(CONFIG['base_path'])

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)

print("Classes mapping:", train_ds.class_to_idx)

# ---------------- MODEL ----------------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ Works for all versions
        self.mean = IMAGENET_MEAN
        self.std  = IMAGENET_STD

        self.base = models.resnet18(pretrained=True)

        for name, p in self.base.named_parameters():
            if not (name.startswith('layer4') or name.startswith('fc')):
                p.requires_grad = False

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

model = CNNModel().to(device)

# ---------------- LOSS with class weighting ----------------
labels_train = [y for _, y in train_ds.samples]
pos = sum(1 for v in labels_train if v == 1)
neg = len(labels_train) - pos
pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler = GradScaler(enabled=CONFIG['use_mixed_precision'])


# ---------------- UTILS ----------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience, self.min_delta = patience, min_delta
        self.best = None
        self.count = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True


def forward_pass(model, loader, collect_probs=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            loss_sum += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if collect_probs:
                all_probs.append(probs.squeeze(1).cpu().numpy())
                all_labels.append(labels.squeeze(1).cpu().numpy())

    acc = 100.0 * correct / max(total, 1)
    avg_loss = loss_sum / max(len(loader), 1)

    if collect_probs:
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        return avg_loss, acc, all_probs, all_labels

    return avg_loss, acc


def find_best_threshold(y_true, y_prob, metric='youden'):
    if metric == 'youden':
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        best = thr[np.argmax(j)]
    else:
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        thr = np.append(thr, 1.0)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best = thr[np.argmax(f1)]

    return float(np.clip(best, 0, 1))


# ---------------- TRAIN ----------------
def train():
    early = EarlyStopping(CONFIG['early_stopping_patience'])
    best_state, best_val_acc = None, -1.0

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']} Training"):
          imgs = imgs.to(device)
          labels = labels.float().unsqueeze(1).to(device)
          
          optimizer.zero_grad(set_to_none=True)
          with autocast(enabled=CONFIG['use_mixed_precision']):
              logits = model(imgs)
              loss = criterion(logits, labels)
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update()
              run_loss += loss.item()
              with torch.no_grad():
                  probs = torch.sigmoid(logits)
                  preds = (probs > 0.5).float()
                  correct += (preds == labels).sum().item()
                  total += labels.size(0)

        train_loss = run_loss / len(train_loader)
        train_acc  = 100 * correct / total

        val_loss, val_acc, val_probs, val_labels = forward_pass(model, valid_loader, collect_probs=True)
        scheduler.step(val_loss)

        print(f"[{epoch}/{CONFIG['num_epochs']}] Train {train_loss:.4f}/{train_acc:.2f}% | "
              f"Valid {val_loss:.4f}/{val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        early(val_loss)
        if early.stop:
            print("Early stopping triggered.")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _, _, val_probs, val_labels = forward_pass(model, valid_loader, collect_probs=True)
    best_thr = find_best_threshold(val_labels, val_probs, metric=CONFIG['threshold_metric'])

    print(f"Best threshold: {best_thr:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': best_thr,
        'class_to_idx': train_ds.class_to_idx,
        'config': CONFIG,
    }, f"{CONFIG['model_name']}_best.pth")

    return best_thr


# ---------------- EVALUATE ----------------
def evaluate(threshold):
    _, _, probs, labels = forward_pass(model, test_loader, collect_probs=True)
    preds = (probs > threshold).astype(int)

    print("\nTEST REPORT")
    print(classification_report(labels, preds, target_names=["Fake","Real"], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    thr = train()
    evaluate(thr)
