#!/usr/bin/env python3
# svm_http_baseline.py

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

# ==================== 参数 ====================
TRAIN_PATH = "/data/wry/基线/train_final.csv"
TEST_PATH  = "/data/wry/统一测试数据/test_2000.csv"
MAL_PATH   = "/data/wry/统一测试数据/Extend_XSS_500.csv"
SAVE_DIR   = "/data/wry/基线"
os.makedirs(SAVE_DIR, exist_ok=True)

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

# ==================== 评估函数 ====================
def evaluate_dataset(model, vectorizer, path, desc="测试集"):
    log(f"加载 {desc} 数据...")
    texts, labels = load_dataset(path)

    X = vectorizer.transform(texts)

    log(f"{desc} 测试中...")

    # SVM 概率（来自 calibration）
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = labels

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
    except Exception as e:
        log(f"AUC计算失败: {e}")

# ==================== 主流程 ====================
log("加载数据...")
train_texts, y_train = load_dataset(TRAIN_PATH)
test_texts, y_test   = load_dataset(TEST_PATH)

log(f"训练集: {len(train_texts)} 测试集: {len(test_texts)}")

# ==================== TF-IDF ====================
log("TF-IDF 向量化...")
vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM,
    analyzer="char",
    lowercase=False
)
vectorizer.fit(train_texts)

X_train = vectorizer.transform(train_texts)

# ==================== SVM 模型 ====================
log("训练 Linear SVM（带概率校准）...")

# 基础 SVM（线性核）
base_svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=5000
)

# 概率校准（用于 AUC / PR-AUC）
model = CalibratedClassifierCV(
    base_svm,
    method="sigmoid",   # Platt scaling（论文常用）
    cv=5
)

model.fit(X_train, y_train)

log("训练完成")

# ==================== 双测试 ====================
evaluate_dataset(model, vectorizer, TEST_PATH, desc="统一测试集")
evaluate_dataset(model, vectorizer, MAL_PATH,  desc="纯恶意样本集")

# ==================== 保存 ====================
joblib.dump(model, os.path.join(SAVE_DIR, "svm_model.pkl"))
joblib.dump(vectorizer, os.path.join(SAVE_DIR, "tfidf_svm.pkl"))

log("完成")