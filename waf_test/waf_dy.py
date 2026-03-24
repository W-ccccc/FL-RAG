import requests
import csv
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

WAF_URL = "http://127.0.0.1:8080"
TIMEOUT = 5


# ======================
# 1. 解析 headers
# ======================
def parse_headers(header_str):
    headers = {}
    if not header_str:
        return headers

    for line in header_str.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()

    return headers


# ======================
# 2. 发送请求
# ======================
def send_request(row):
    method = row["method"]
    uri = row["uri"]
    query = row["query"]
    body = row["body"]

    url = f"{WAF_URL}{uri}"
    if query:
        url += "?" + query

    headers = parse_headers(row["headers"])

    try:
        r = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            timeout=TIMEOUT
        )

        return r.status_code

    except Exception as e:
        return 0  # 请求失败


# ======================
# 3. 判定逻辑（核心）
# ======================
def waf_predict(status_code):
    # CRS默认拦截返回 403 / 406
    if status_code in [403, 406]:
        return "malicious"
    return "benign"


# ======================
# 4. 主评测流程
# ======================
def evaluate(csv_path, output_path=r"C:\Users\WRY\Desktop\waf测试\result.csv"):
    y_true = []
    y_pred = []

    results = []

    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            status = send_request(row)
            pred = waf_predict(status)

            y_true.append(row["type"])
            y_pred.append(pred)

            results.append({
                "id": i,
                "true": row["type"],
                "pred": pred,
                "status": status
            })

            print(f"[{i}] status={status} pred={pred}")

            time.sleep(0.01)  # 防止WAF压力过大

    # ======================
    # 5. 计算指标
    # ======================
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="malicious")
    recall = recall_score(y_true, y_pred, pos_label="malicious")
    f1 = f1_score(y_true, y_pred, pos_label="malicious")

    print("\n===== Evaluation Result =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ======================
    # 6. 保存结果
    # ======================
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "true", "pred", "status"])
        writer.writeheader()
        writer.writerows(results)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ======================
# 7. 入口
# ======================
if __name__ == "__main__":
    evaluate(r"C:\Users\WRY\Desktop\waf测试\test_2000.csv")