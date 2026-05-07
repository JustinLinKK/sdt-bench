import os, json, random, time, math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoModel

# -------------------- Reproducibility --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------- Paths --------------------
INPUT_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
LABEL_MAP_JSON = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# -------------------- Load metadata --------------------
train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)
with open(LABEL_MAP_JSON, "r") as f:
    label_map = json.load(f)
NUM_CLASSES = int(train_df["label"].nunique())  # 5

# -------------------- Cleaning --------------------
train_df = train_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
existing_mask = train_df["image_id"].apply(
    lambda x: os.path.exists(os.path.join(TRAIN_IMG_DIR, x))
)
train_df = train_df[existing_mask].reset_index(drop=True)

# -------------------- Class weights --------------------
class_counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
class_weights = (class_counts.sum() / (NUM_CLASSES * class_counts)).astype(np.float32)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# -------------------- Stratified split --------------------
train_split, val_split = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"].values
)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)

# -------------------- Image params --------------------
IMG_SIZE = 256
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# -------------------- Transforms --------------------
train_transform = transforms.Compose(
    [
        transforms.Resize(
            (int(IMG_SIZE * 1.15), int(IMG_SIZE * 1.15)),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.7, 1.0),
            ratio=(0.85, 1.15),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply(
            [
                transforms.RandomRotation(
                    20, interpolation=transforms.InterpolationMode.BICUBIC
                )
            ],
            p=0.5,
        ),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
                )
            ],
            p=0.7,
        ),
        transforms.RandAugment(num_ops=2, magnitude=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0.0
        ),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ]
)


# -------------------- Datasets --------------------
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

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
        if self.has_label:
            label = int(row["label"])
            return img, label
        else:
            return img, row["image_id"]


train_dataset = CassavaDataset(
    train_split, TRAIN_IMG_DIR, transform=train_transform, has_label=True
)
val_dataset = CassavaDataset(
    val_split, TRAIN_IMG_DIR, transform=eval_transform, has_label=True
)
test_dataset = CassavaDataset(
    sub_df, TEST_IMG_DIR, transform=eval_transform, has_label=False
)

# -------------------- DataLoaders --------------------
# NUM_WORKERS=0 to avoid Python 3.14 forkserver ConnectionResetError
# (forkserver fails to connect to new worker processes in this environment).
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 16
NUM_WORKERS = 0

sample_weights = train_split["label"].map(lambda y: float(class_weights[int(y)])).values
sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=(NUM_WORKERS > 0),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_EVAL,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_EVAL,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)

# Sanity check: confirm DataLoader can iterate without spawning workers
try:
    _probe_iter = iter(val_loader)
    _probe_batch = next(_probe_iter)
    del _probe_iter, _probe_batch
except Exception as _e:
    print(f"DataLoader probe warning: {_e}")

print(
    f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} | Classes: {NUM_CLASSES} | ImgSize: {IMG_SIZE}"
)
print(f"Class counts (sorted): {class_counts.tolist()}")
print(f"Class weights: {class_weights.tolist()}")


# -------------------- Position embedding interpolation utility --------------------
def interpolate_siglip_pos_embed(vision_model, new_image_size: int):
    embeddings = vision_model.embeddings
    patch_size = embeddings.patch_size
    old_num_positions = embeddings.position_embedding.num_embeddings
    old_grid = int(round(old_num_positions**0.5))
    new_grid = new_image_size // patch_size
    new_num_positions = new_grid * new_grid

    if new_num_positions == old_num_positions:
        return

    old_weight = embeddings.position_embedding.weight.data
    embed_dim = old_weight.shape[1]

    old_2d = (
        old_weight.reshape(old_grid, old_grid, embed_dim)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    new_2d = F.interpolate(
        old_2d, size=(new_grid, new_grid), mode="bicubic", align_corners=False
    )
    new_weight = (
        new_2d.squeeze(0).permute(1, 2, 0).reshape(new_num_positions, embed_dim)
    ).contiguous()

    new_pos_emb = nn.Embedding(new_num_positions, embed_dim)
    new_pos_emb.weight.data.copy_(new_weight.to(old_weight.dtype))
    embeddings.position_embedding = new_pos_emb

    embeddings.num_patches = new_num_positions
    embeddings.num_positions = new_num_positions
    embeddings.image_size = new_image_size
    if hasattr(embeddings, "position_ids"):
        embeddings.register_buffer(
            "position_ids",
            torch.arange(new_num_positions).expand((1, -1)),
            persistent=False,
        )


# -------------------- SigLIP2 classification model --------------------
class CassavaSiglip2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        img_size: int = 384,
        dropout: float = 0.30,
        hidden_dim: int = 512,
        model_name: str = "google/siglip2-large-patch16-256",
    ):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name)
        interpolate_siglip_pos_embed(backbone.vision_model, img_size)
        self.backbone = backbone
        feat_dim = 1024

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pixel_values):
        feats = self.backbone.get_image_features(pixel_values=pixel_values)
        feats = F.normalize(feats, dim=-1) * (feats.shape[-1] ** 0.5)
        logits = self.head(feats)
        return logits


# -------------------- Instantiate model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CassavaSiglip2Classifier(
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
    dropout=0.30,
    hidden_dim=512,
    model_name="google/siglip2-large-patch16-256",
).to(device)

# Enable gradient checkpointing on the backbone to reduce activation memory.
try:
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
    elif hasattr(model.backbone, "vision_model") and hasattr(
        model.backbone.vision_model, "gradient_checkpointing_enable"
    ):
        model.backbone.vision_model.gradient_checkpointing_enable()
except Exception as _e:
    print(f"Gradient checkpointing not enabled: {_e}")

# Reduce fragmentation risk
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.cuda.empty_cache()


# -------------------- Loss --------------------
criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor.to(device),
    label_smoothing=0.1,
)


# -------------------- Optimizer --------------------
BACKBONE_LR = 1e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.05


def _param_groups(module, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "embedding" in name.lower()
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


param_groups = _param_groups(model.backbone, BACKBONE_LR, WEIGHT_DECAY) + _param_groups(
    model.head, HEAD_LR, WEIGHT_DECAY
)

optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

scaler = torch.cuda.amp.GradScaler()

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"Model: SigLIP2-large | params={n_params/1e6:.1f}M | img_size={IMG_SIZE} | "
    f"backbone_lr={BACKBONE_LR} | head_lr={HEAD_LR} | wd={WEIGHT_DECAY}"
)

# ----------------- Training config -----------------
EPOCHS = 4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0
BEST_CKPT = "./working/best_siglip2.pt"

steps_per_epoch = max(1, len(train_loader))
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch

base_lrs = [g["lr"] for g in optimizer.param_groups]


def set_lr(global_step):
    if global_step < warmup_steps:
        scale = float(global_step + 1) / float(max(1, warmup_steps))
    else:
        progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scale = max(scale, 0.01)
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = base * scale


# ----------------- TTA inference -----------------
@torch.no_grad()
def tta_softmax(model, pixel_values):
    model.eval()
    views = [
        pixel_values,
        torch.flip(pixel_values, dims=[3]),
        torch.flip(pixel_values, dims=[2]),
    ]
    probs = None
    for v in views:
        with torch.cuda.amp.autocast():
            logits = model(v)
        p = F.softmax(logits.float(), dim=-1)
        probs = p if probs is None else probs + p
    return probs / len(views)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = tta_softmax(model, imgs)
        preds = probs.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return accuracy_score(all_labels, all_preds), all_preds, all_labels


# ----------------- Training loop -----------------
best_val_acc = -1.0
best_state = None
global_step = 0
train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_correct, running_n = 0.0, 0, 0
    epoch_start = time.time()

    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        set_lr(global_step)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (logits.argmax(-1) == labels).sum().item()
        running_n += imgs.size(0)
        global_step += 1

    train_loss = running_loss / max(1, running_n)
    train_acc = running_correct / max(1, running_n)

    val_acc, _, _ = evaluate(model, val_loader)
    elapsed = time.time() - epoch_start
    print(
        f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
        f"val_acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        torch.save(best_state, BEST_CKPT)

# ----------------- Load best & final validation -----------------
if best_state is not None:
    model.load_state_dict(best_state)
elif os.path.exists(BEST_CKPT):
    model.load_state_dict(torch.load(BEST_CKPT, map_location=device))

final_val_acc, _, _ = evaluate(model, val_loader)

# ----------------- Test inference -----------------
model.eval()
test_preds, test_ids = [], []
with torch.no_grad():
    for imgs, image_ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = tta_softmax(model, imgs)
        preds = probs.argmax(dim=-1).cpu().numpy().tolist()
        test_preds.extend(preds)
        if isinstance(image_ids, (list, tuple)):
            test_ids.extend(list(image_ids))
        else:
            test_ids.extend(list(image_ids))

submission = pd.DataFrame({"image_id": test_ids, "label": test_preds})
submission = (
    sub_df[["image_id"]]
    .merge(submission, on="image_id", how="left")
    .fillna({"label": 4})
)
submission["label"] = submission["label"].astype(int)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)

total_time = time.time() - train_start
print(
    f"Training complete in {total_time/60:.1f} min | best_val_acc={best_val_acc:.4f} | "
    f"final_val_acc={final_val_acc:.4f} | submission rows={len(submission)}"
)

score = float(final_val_acc)
print(f"Final Validation Score: {score}")
