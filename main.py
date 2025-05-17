# text_classification_pipeline.py
import os, random, matplotlib, pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import Text2Class

matplotlib.use("TkAgg")

CSV_PATH            = "final_dataset.csv"
MODEL_NAME_TOKENIZER = "prajjwal1/bert-tiny"
MODEL_PATH           = Path("model.pt")

BATCH_SIZE  = 16
NUM_EPOCHS  = 100
LR          = 2e-5
MAX_LEN     = 128
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESUME_TRAINING = False
LOAD_ONLY       = False
FORCE_TRAIN     = True
# ───────────────────────────────────────────────────────────────────────────────
if RESUME_TRAINING and LOAD_ONLY:
    raise ValueError("RESUME_TRAINING and LOAD_ONLY cannot both be True")
# ───────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain `text` and `label` columns.")
    return df


def build_label_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniques = sorted(df["label"].unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(uniques)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_dataloaders(df: pd.DataFrame, label2id: Dict[str, int], tokenizer):
    df["label_id"] = df["label"].map(label2id)
    train_df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=0.8, random_state=SEED)
    )
    test_df = df.drop(train_df.index)

    train_ds = TextDataset(train_df["text"].tolist(), train_df["label_id"].tolist())
    test_ds = TextDataset(test_df["text"].tolist(), test_df["label_id"].tolist())

    def collate_fn(batch):
        texts, y = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {**enc, "labels": torch.tensor(y)}

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, running = 0, 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = criterion(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]),
            batch["labels"],
        )
        loss.backward()
        optimizer.step()
        running += loss.item() * batch["labels"].size(0)
        total += batch["labels"].size(0)
    return running / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, running = 0, 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
        loss = criterion(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]),
            batch["labels"],
        )
        running += loss.item() * batch["labels"].size(0)
        total += batch["labels"].size(0)
    return running / total


def plot_loss(train_hist, test_hist, out_png="loss_curve.png"):
    plt.figure()
    plt.plot(train_hist, label="Train loss")
    plt.plot(test_hist, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training vs. Test loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()


def main():
    set_seed()
    df = load_dataframe(CSV_PATH)
    label2id, _ = build_label_maps(df)
    num_classes = len(label2id)
    print(f"🏷  Detected {num_classes} unique labels.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TOKENIZER)
    model = Text2Class(
        num_classes,
        pretrained_model_name="custom",
        tokenizer_vocab_size=len(tokenizer),
    ).to(DEVICE)

    checkpoint_found = MODEL_PATH.exists()

    # ─── load weights only if explicitly requested ───
    if checkpoint_found and (RESUME_TRAINING or LOAD_ONLY):
        print(f"🔄  Loading weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    elif checkpoint_found:
        print("ℹ️  Checkpoint found but ignored (fresh training requested).")
    # ------------------------------------------------

    if LOAD_ONLY:
        if not checkpoint_found:
            print(f"❌  No checkpoint found at {MODEL_PATH}.")
        else:
            print("✅  Model loaded; skipping training as requested.")
        return

    if not checkpoint_found and not RESUME_TRAINING and not FORCE_TRAIN:
        print("ℹ️  Neither checkpoint found nor training requested. Exiting.")
        return

    # ---- regular training loop ----
    train_loader, test_loader = make_dataloaders(df, label2id, tokenizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    train_hist, test_hist = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss  = evaluate(model, test_loader, criterion)
        train_hist.append(train_loss)
        test_hist.append(test_loss)
        print(f"[Epoch {epoch:02d}]  train = {train_loss:.4f} | test = {test_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾  Saved model to {MODEL_PATH}")
    plot_loss(train_hist, test_hist)


if __name__ == "__main__":
    main()