import os
import json
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Paths & constants
# -------------------------
INPUT_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

IMG_SIZE = 256  # multiple of 16 -> compatible with DINOv3 patch-16
NUM_CLASSES = 5
SEED = 42
VAL_FRAC = 0.15
BATCH_SIZE = 32
NUM_WORKERS = 4

np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Load metadata
# -------------------------
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json"), "r") as f:
    label_map = json.load(f)

# Drop rows whose image file is missing (data cleaning)
existing_mask = train_df["image_id"].apply(
    lambda x: os.path.isfile(os.path.join(TRAIN_IMG_DIR, x))
)
train_df = train_df[existing_mask].reset_index(drop=True)

# Test image list: prefer sample_submission order so prediction order matches
test_image_ids = sample_sub["image_id"].tolist()
test_image_ids = [
    fn for fn in test_image_ids if os.path.isfile(os.path.join(TEST_IMG_DIR, fn))
]

# -------------------------
# Stratified train/validation split (split FIRST -> no leakage)
# -------------------------
trn_df, val_df = train_test_split(
    train_df,
    test_size=VAL_FRAC,
    random_state=SEED,
    stratify=train_df["label"].values,
)
trn_df = trn_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Class weights computed ONLY from training fold
train_class_counts = (
    trn_df["label"]
    .value_counts()
    .sort_index()
    .reindex(range(NUM_CLASSES), fill_value=0)
    .values.astype(np.float32)
)
class_weights = train_class_counts.sum() / (
    NUM_CLASSES * np.maximum(train_class_counts, 1.0)
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# -------------------------
# Image transforms
# -------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05
        ),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# -------------------------
# Dataset class
# -------------------------
class CassavaDataset(Dataset):
    def __init__(self, image_ids, labels, image_dir, transform):
        self.image_ids = list(image_ids)
        self.labels = None if labels is None else list(labels)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def _load(self, idx):
        path = os.path.join(self.image_dir, self.image_ids[idx])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        return img

    def __getitem__(self, idx):
        img = self._load(idx)
        img = self.transform(img)
        if self.labels is None:
            return img, self.image_ids[idx]
        return img, int(self.labels[idx])


train_dataset = CassavaDataset(
    image_ids=trn_df["image_id"].values,
    labels=trn_df["label"].values,
    image_dir=TRAIN_IMG_DIR,
    transform=train_transform,
)
val_dataset = CassavaDataset(
    image_ids=val_df["image_id"].values,
    labels=val_df["label"].values,
    image_dir=TRAIN_IMG_DIR,
    transform=eval_transform,
)
test_dataset = CassavaDataset(
    image_ids=test_image_ids,
    labels=None,
    image_dir=TEST_IMG_DIR,
    transform=eval_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

print(
    f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}"
)
print(f"Train class counts: {train_class_counts.tolist()}")
print(f"Class weights: {class_weights.tolist()}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE} | Num classes: {NUM_CLASSES}")

# -------------------------
# Pretrained backbone via timm (with robust fallbacks)
# -------------------------
import timm

BACKBONE_CANDIDATES = [
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "tf_efficientnet_b3.ns_jft_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "resnet50",
]

backbone = None
loaded_name = None
for _name in BACKBONE_CANDIDATES:
    try:
        backbone = timm.create_model(
            _name, pretrained=True, num_classes=0, global_pool="avg"
        )
        loaded_name = _name
        print(f"Loaded backbone via timm: {_name}")
        break
    except Exception as _e:
        print(f"Failed to load {_name}: {repr(_e)}")

if backbone is None:
    # Last-resort fallback: torchvision ResNet50
    from torchvision.models import resnet50, ResNet50_Weights
    try:
        tv_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        tv_model = resnet50(weights=None)
    feat_dim_tv = tv_model.fc.in_features
    tv_model.fc = nn.Identity()
    backbone = tv_model
    backbone.num_features = feat_dim_tv
    loaded_name = "torchvision_resnet50"
    print("Loaded backbone via torchvision: resnet50")

FEAT_DIM = int(getattr(backbone, "num_features", 2048))


class CassavaClassifier(nn.Module):
    def __init__(
        self,
        backbone,
        feat_dim,
        num_classes=5,
        dropout=0.3,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        feat = self.norm(feat)
        logits = self.head(feat)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CassavaClassifier(
    backbone=backbone,
    feat_dim=FEAT_DIM,
    num_classes=NUM_CLASSES,
    dropout=0.3,
    freeze_backbone=False,
).to(device)


# -------------------------
# Loss: Label-smoothing CE with class weights
# -------------------------
class LabelSmoothingWeightedCE(nn.Module):
    def __init__(
        self, weight: torch.Tensor, smoothing: float = 0.1, num_classes: int = 5
    ):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.register_buffer("weight", weight.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs, self.smoothing / (self.num_classes - 1)
            )
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        per_class = -(true_dist * log_probs) * self.weight.unsqueeze(0)
        per_sample = per_class.sum(dim=-1)
        sample_w = self.weight[target]
        return per_sample.sum() / sample_w.sum().clamp_min(1e-8)


criterion = LabelSmoothingWeightedCE(
    weight=class_weights.to(device),
    smoothing=0.1,
    num_classes=NUM_CLASSES,
).to(device)


# -------------------------
# Optimizer: AdamW with differential learning rates
# -------------------------
BACKBONE_LR = 1e-4
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

NUM_EPOCHS = 6
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
scaler = torch.cuda.amp.GradScaler()

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"Model: {loaded_name} (feat_dim={FEAT_DIM}) + MLP head | params total={n_total/1e6:.1f}M trainable={n_train/1e6:.1f}M"
)
print(
    f"Optimizer: AdamW | backbone_lr={BACKBONE_LR} head_lr={HEAD_LR} wd={WEIGHT_DECAY}"
)
print(
    f"Loss: LabelSmoothingWeightedCE (smoothing=0.1) | Scheduler: CosineAnnealing(T_max={NUM_EPOCHS})"
)

# -------------------------
# Training loop with mixed precision + best-checkpoint saving
# -------------------------
BEST_CKPT_PATH = os.path.join(WORKING_DIR, "best_model.pt")
GRAD_CLIP = 1.0

best_val_acc = -1.0
best_epoch = -1
best_state = None

start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
        )
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            running_total += targets.size(0)
            running_loss += loss.item() * targets.size(0)

    train_loss = running_loss / max(running_total, 1)
    train_acc = running_correct / max(running_total, 1)

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            val_correct += (preds == targets).sum().item()
            val_total += targets.size(0)
            val_loss_sum += loss.item() * targets.size(0)

    val_loss = val_loss_sum / max(val_total, 1)
    val_acc = val_correct / max(val_total, 1)

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, BEST_CKPT_PATH)

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | best_val_acc={best_val_acc:.4f} (ep {best_epoch}) | elapsed={elapsed:.0f}s"
    )

# Load best checkpoint
if best_state is not None:
    model.load_state_dict(best_state)
elif os.path.isfile(BEST_CKPT_PATH):
    model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device))
model.eval()


# -------------------------
# TTA inference helper
# -------------------------
@torch.no_grad()
def predict_with_tta(loader, has_labels: bool):
    all_probs = []
    all_ids = []
    all_labels = []
    for batch in loader:
        if has_labels:
            images, targets = batch
        else:
            images, ids = batch
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits1 = model(images)
            logits2 = model(torch.flip(images, dims=[3]))
        probs = (
            F.softmax(logits1.float(), dim=1) + F.softmax(logits2.float(), dim=1)
        ) / 2.0
        all_probs.append(probs.cpu().numpy())

        if has_labels:
            all_labels.extend(
                targets.numpy().tolist()
                if isinstance(targets, torch.Tensor)
                else list(targets)
            )
        else:
            all_ids.extend(list(ids))

    probs_np = np.concatenate(all_probs, axis=0)
    return (
        probs_np,
        (all_ids if not has_labels else None),
        (all_labels if has_labels else None),
    )


# Final validation metric
val_probs, _, val_labels = predict_with_tta(val_loader, has_labels=True)
val_preds = val_probs.argmax(axis=1)
val_labels_np = np.array(val_labels, dtype=np.int64)

final_val_accuracy = accuracy_score(val_labels_np, val_preds)
try:
    print("Per-class report on validation:")
    print(
        classification_report(
            val_labels_np,
            val_preds,
            labels=list(range(NUM_CLASSES)),
            digits=4,
            zero_division=0,
        )
    )
except Exception:
    pass

# Test inference -> submission.csv
test_probs, test_ids, _ = predict_with_tta(test_loader, has_labels=False)
test_preds = test_probs.argmax(axis=1).astype(int)

pred_df = pd.DataFrame({"image_id": test_ids, "label": test_preds})

majority_class = int(trn_df["label"].value_counts().idxmax())
sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
sub["label"] = sub["label"].fillna(majority_class).astype(int)

SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
sub.to_csv(SUBMISSION_PATH, index=False)

print(
    f"Saved submission to {SUBMISSION_PATH} | rows={len(sub)} | best_epoch={best_epoch} | best_val_acc={best_val_acc:.4f}"
)
print(f"Final Validation Score: {final_val_accuracy}")
