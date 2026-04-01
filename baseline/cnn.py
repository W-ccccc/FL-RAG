#!/usr/bin/env python3
# textcnn_http_dual_test.py

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==================== 参数 ====================
TRAIN_PATH = "/data/wry/基线/train_final.csv"
TEST_PATH  = "/data/wry/统一测试数据/test_2000.csv"
MAL_PATH   = "/data/wry/统一测试数据/Extend_XSS_500.csv"
SAVE_DIR   = "/data/wry/基线"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
PATIENCE = 5

TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM = (1, 4)

# ==================== 工具函数 ====================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def build_text(row):
    return (
        f"[METHOD] {row['method']} "
        f"[URL] {row['url']} "
        f"[URI] {row['uri']} "
        f"[QUERY] {row['query']} "
        f"[HEADERS] {row['headers']} "
        f"[BODY] {row['body']}"
    )

def load_dataset(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    texts = df.apply(build_text, axis=1)
    labels = df["type"].str.lower().apply(lambda x: 0 if x=="benign" else 1).values
    return texts.tolist(), labels

def evaluate_dataset(model, vectorizer, path, desc="测试集"):
    log(f"加载 {desc} 数据...")
    texts, labels = load_dataset(path)

    X = vectorizer.transform(texts).toarray()
    X = np.expand_dims(X, 2)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE)

    log(f"{desc} 测试中...")
    model.eval()
    probs, truths = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            p = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(p)
            truths.extend(yb.numpy().flatten())

    y_prob = np.array(probs)
    y_true = np.array(truths)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = np.mean(y_pred == y_true)
    log(f"{desc} Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"{desc} Confusion Matrix:\n", cm)

    print(f"\n{desc} Classification Report:\n")
    print(classification_report(y_true, y_pred))

    try:
        rocauc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        log(f"{desc} ROC-AUC: {rocauc:.4f} PR-AUC: {pr_auc:.4f}")
    except:
        pass

# ==================== 主流程 ====================
log("加载数据...")
train_texts, y_train = load_dataset(TRAIN_PATH)
test_texts, y_test   = load_dataset(TEST_PATH)

log(f"训练集: {len(train_texts)} 测试集: {len(test_texts)}")

# TF-IDF
log("TF-IDF 向量化...")
vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM,
    analyzer="char",
    lowercase=False
)
vectorizer.fit(train_texts)

X_train = vectorizer.transform(train_texts).toarray()
X_train = np.expand_dims(X_train, 2)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

# ==================== TextCNN ====================
class TextCNN(nn.Module):
    def __init__(self, input_dim, num_filters=100):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, k)
            for k in [3,4,5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * 3, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 1, seq_len)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max(c, dim=2)[0]
            conv_outs.append(p)
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

model = TextCNN(input_dim=1).to(DEVICE)

# 类不平衡处理
pos = np.sum(y_train==1)
neg = np.sum(y_train==0)
pos_weight = neg / (pos + 1e-12)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==================== 训练 ====================
log("开始训练...")
best_loss = float("inf")
patience_cnt = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    log(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_cnt = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            log("Early stopping")
            break

# ==================== 双测试 ====================
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))

evaluate_dataset(model, vectorizer, TEST_PATH, desc="统一测试集")
evaluate_dataset(model, vectorizer, MAL_PATH,  desc="纯恶意样本集")

# ==================== 保存 ====================
joblib.dump(vectorizer, os.path.join(SAVE_DIR, "tfidf.pkl"))
log("完成")
