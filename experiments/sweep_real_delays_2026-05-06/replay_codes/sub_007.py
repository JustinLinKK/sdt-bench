import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast
try:
    # Newer torch API
    from torch.amp import GradScaler as _NewGradScaler
    def GradScaler():
        return _NewGradScaler('cuda')
except Exception:
    from torch.cuda.amp import GradScaler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

IMG_SIZE = 256  # multiple of patch size 16 for DINOv3
NUM_CLASSES = 5
SEED = 42
BATCH_SIZE = 32
# Use 0 workers to avoid forkserver ConnectionResetError on this environment.
# Single-process data loading is reliable and fits within the time budget.
NUM_WORKERS = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Load metadata
# -----------------------------------------------------------------------------
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json"), "r") as f:
    label_map = json.load(f)

# Drop any rows whose image file is missing (defensive cleaning)
train_df = train_df[
    train_df["image_id"].apply(lambda x: os.path.exists(os.path.join(TRAIN_IMG_DIR, x)))
].reset_index(drop=True)

# Build test_df from the actual files in test_images (handles full hidden test set)
test_image_files = sorted(os.listdir(TEST_IMG_DIR))
test_df = pd.DataFrame({"image_id": test_image_files})
if set(sample_sub["image_id"].tolist()).issubset(set(test_image_files)):
    test_df = sample_sub[["image_id"]].copy()

# -----------------------------------------------------------------------------
# Stratified split (train / val) — 90/10 to maximize training data
# -----------------------------------------------------------------------------
train_split_df, val_split_df = train_test_split(
    train_df,
    test_size=0.10,
    stratify=train_df["label"],
    random_state=SEED,
)
train_split_df = train_split_df.reset_index(drop=True)
val_split_df = val_split_df.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Class weights & sampler (combat severe label-3 imbalance)
# -----------------------------------------------------------------------------
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_split_df["label"].values,
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

sample_weights = (
    train_split_df["label"]
    .map({c: class_weights_np[c] for c in range(NUM_CLASSES)})
    .values
)
sample_weights = torch.tensor(sample_weights, dtype=torch.double)
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# -----------------------------------------------------------------------------
# Transforms — DINOv3 ImageNet normalization
# -----------------------------------------------------------------------------
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose(
    [
        transforms.Resize((int(IMG_SIZE * 1.15), int(IMG_SIZE * 1.15))),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ]
)

tta_transforms = [
    eval_transform,
    transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
        ]
    ),
    transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
        ]
    ),
]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        if self.has_labels:
            label = int(row["label"])
            return img, label
        return img, row["image_id"]


# -----------------------------------------------------------------------------
# Build datasets & loaders
# -----------------------------------------------------------------------------
train_dataset = CassavaDataset(
    train_split_df, TRAIN_IMG_DIR, train_transform, has_labels=True
)
val_dataset = CassavaDataset(
    val_split_df, TRAIN_IMG_DIR, eval_transform, has_labels=True
)
test_dataset = CassavaDataset(test_df, TEST_IMG_DIR, eval_transform, has_labels=False)

# NOTE: Use natural class distribution (matches val/test) instead of
# WeightedRandomSampler. Combined with unweighted CE this avoids the
# double-balancing pitfall that crushed val accuracy below the trivial baseline.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=(NUM_WORKERS > 0),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)

print(
    f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
)
print(f"Class weights: {class_weights_np.round(3).tolist()}")
print(f"Image size: {IMG_SIZE} | Classes: {NUM_CLASSES}")

# -----------------------------------------------------------------------------
# Backbone via timm (DINOv3 local hub was unavailable — use robust fallback)
# -----------------------------------------------------------------------------
import timm


# -----------------------------------------------------------------------------
# Classifier — timm pretrained backbone + small MLP head
# Keeping class name `DINOv3Classifier` so downstream code (optimizer param
# grouping by 'backbone.' prefix, instantiation call) remains unchanged.
# -----------------------------------------------------------------------------
class DINOv3Classifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3, hidden_dim=512):
        super().__init__()
        # Try a list of progressively simpler / more available pretrained models.
        candidate_models = [
            "tf_efficientnet_b4.ns_jft_in1k",
            "tf_efficientnet_b4",
            "tf_efficientnet_b3.ns_jft_in1k",
            "tf_efficientnet_b3",
            "efficientnet_b3",
            "convnext_small.fb_in22k_ft_in1k",
            "convnext_small",
            "resnet50",
        ]
        self.backbone = None
        loaded_name = None
        last_err = None
        for mname in candidate_models:
            try:
                self.backbone = timm.create_model(
                    mname, pretrained=True, num_classes=0, global_pool="avg"
                )
                loaded_name = mname
                break
            except Exception as e:
                last_err = e
                continue
        if self.backbone is None:
            # Final fallback: non-pretrained resnet50 (still trainable)
            self.backbone = timm.create_model(
                "resnet50", pretrained=False, num_classes=0, global_pool="avg"
            )
            loaded_name = "resnet50 (no pretrained)"
        print(f"Loaded backbone: {loaded_name}")

        self.feat_dim = int(self.backbone.num_features)

        self.norm = nn.LayerNorm(self.feat_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.feat_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def extract_features(self, x):
        # timm backbones with num_classes=0 + global_pool='avg' return [B, feat_dim]
        feats = self.backbone(x)
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        h = self.norm(feats)
        h = self.dropout1(h)
        h = self.act(self.fc1(h))
        h = self.dropout2(h)
        logits = self.fc2(h)
        return logits


# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DINOv3Classifier(num_classes=NUM_CLASSES, dropout=0.3, hidden_dim=512).to(
    DEVICE
)

# -----------------------------------------------------------------------------
# Loss: cross-entropy with label smoothing.
# Class weights REMOVED — using natural-distribution sampling instead, since
# accuracy is the metric and the val/test sets follow the natural distribution.
# -----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1,
)

# -----------------------------------------------------------------------------
# Optimizer: differential learning rates (low for backbone, high for head)
# -----------------------------------------------------------------------------
BACKBONE_LR = 5e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05

backbone_params, head_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith("backbone."):
        backbone_params.append(p)
    else:
        head_params.append(p)

optimizer = AdamW(
    [
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
    ],
    betas=(0.9, 0.999),
    eps=1e-8,
)

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"Model: timm backbone + MLP head | total={n_total/1e6:.1f}M | trainable={n_train/1e6:.1f}M"
)
print(
    f"Optimizer: AdamW | backbone_lr={BACKBONE_LR} | head_lr={HEAD_LR} | wd={WEIGHT_DECAY}"
)

# -----------------------------------------------------------------------------
# Training config
# -----------------------------------------------------------------------------
EPOCHS = 12
GRAD_CLIP = 1.0
BEST_MODEL_PATH = f"{WORKING_DIR}/best_dinov3_cassava.pt"

steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[BACKBONE_LR, HEAD_LR],
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

scaler = GradScaler()


# -----------------------------------------------------------------------------
# Validation helper (single-pass, no TTA — used for monitoring)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    return total_loss / max(1, n), acc


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
best_val_acc = -1.0
best_epoch = -1
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_correct, running_n = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_n += imgs.size(0)
            running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / max(1, running_n)
    train_acc = running_correct / max(1, running_n)
    val_loss, val_acc = evaluate(model, val_loader, DEVICE)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(
            {"model_state_dict": model.state_dict(), "val_acc": best_val_acc},
            BEST_MODEL_PATH,
        )

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
        f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | best={best_val_acc:.4f}@{best_epoch} | {elapsed/60:.1f}min"
    )

# -----------------------------------------------------------------------------
# Load best model
# -----------------------------------------------------------------------------
ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# -----------------------------------------------------------------------------
# TTA-averaged softmax inference (same logic for val & test)
# -----------------------------------------------------------------------------
@torch.no_grad()
def tta_predict_softmax(df, img_dir, has_labels=False):
    n = len(df)
    probs_sum = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    labels_out = None
    if has_labels:
        labels_out = np.zeros(n, dtype=np.int64)

    for t_idx, tfm in enumerate(tta_transforms):
        ds = CassavaDataset(df, img_dir, tfm, has_labels=has_labels)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
        )
        idx = 0
        for batch in loader:
            if has_labels:
                imgs, lbls = batch
            else:
                imgs, _ = batch
            imgs = imgs.to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            bsz = probs.shape[0]
            probs_sum[idx : idx + bsz] += probs
            if has_labels and t_idx == 0:
                labels_out[idx : idx + bsz] = lbls.numpy()
            idx += bsz

    probs_avg = probs_sum / len(tta_transforms)
    return probs_avg, labels_out


# -----------------------------------------------------------------------------
# Validation score with TTA (matches test inference)
# -----------------------------------------------------------------------------
val_probs, val_labels = tta_predict_softmax(
    val_split_df, TRAIN_IMG_DIR, has_labels=True
)
val_preds = val_probs.argmax(axis=1)
final_val_acc = accuracy_score(val_labels, val_preds)

# -----------------------------------------------------------------------------
# Test inference & submission
# -----------------------------------------------------------------------------
test_probs, _ = tta_predict_softmax(test_df, TEST_IMG_DIR, has_labels=False)
test_preds = test_probs.argmax(axis=1).astype(int)

submission = pd.DataFrame({"image_id": test_df["image_id"].values, "label": test_preds})
sub_path = f"{SUBMISSION_DIR}/submission.csv"
submission.to_csv(sub_path, index=False)
print(
    f"Saved submission: {sub_path} | rows={len(submission)} | best_epoch={best_epoch} | best_val_acc(no-TTA)={best_val_acc:.4f}"
)

score = final_val_acc
print(f"Final Validation Score: {score}")
