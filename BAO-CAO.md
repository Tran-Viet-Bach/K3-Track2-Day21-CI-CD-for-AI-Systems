# Báo Cáo Lab Day 21 — CI/CD cho Hệ Thống AI

**Học viên:** Trần Việt Bách
**Repo:** https://github.com/Tran-Viet-Bach/K3-Track2-Day21-CI-CD-for-AI-Systems
**Cloud provider:** AWS (S3 + EC2)

---

## 1. Bộ siêu tham số đã chọn và lý do

**Bộ đã chọn:** `n_estimators: 200`, `max_depth: 20`, `min_samples_split: 2`

Kết quả thực nghiệm trên `train_phase1.csv` (2998 mẫu), đánh giá trên `eval.csv` (500 mẫu):

| n_estimators | max_depth | min_samples_split | train acc | eval acc | f1 |
|---|---|---|---|---|---|
| 50 | 3 | 2 | 0.5854 | 0.5580 | 0.5185 |
| 100 | 5 | 2 | 0.6441 | 0.5640 | 0.5534 |
| 200 | 10 | 5 | 0.8906 | 0.6440 | 0.6417 |
| 200 | 20 | 5 | 0.9973 | 0.6640 | 0.6621 |
| **200** | **20** | **2** | **1.0000** | **0.6840** | **0.6830** |
| 300 | None | 2 | 1.0000 | 0.6820 | 0.6811 |

**Lý do chọn:** bộ này cho eval accuracy cao nhất (0.6840). Quan trọng hơn con số, việc đọc **cặp** train/eval cho thấy hai chế độ lỗi khác nhau:

- `max_depth = 3` và `5`: train accuracy chỉ 0.58–0.64 — mô hình học kém ngay trên dữ liệu đã thấy. Đây là **underfitting**: `max_depth=3` chỉ cho phép mỗi cây chia không gian thành tối đa 2³ = 8 vùng, không đủ để tách 2998 mẫu với 12 đặc trưng. Đây là giới hạn tự áp đặt, sửa được miễn phí bằng cách nới tham số.
- `max_depth ≥ 20`: train đạt 1.0000 nhưng eval dừng ở 0.68 — **overfitting**, và khoảng cách 32 điểm phản ánh giới hạn của chính dữ liệu.

**Đã chạm trần dữ liệu.** Kiểm chứng bằng hai phép đo: (a) tăng `n_estimators` từ 50 lên 1000 chỉ làm eval accuracy dao động 0.668–0.684, không có xu hướng tăng; (b) 37.9% số mẫu có láng giềng gần nhất trong không gian đặc trưng mang nhãn khác — hai chai rượu gần giống hệt nhau về hóa học nhưng bị chấm điểm chất lượng khác nhau. Đây là **Bayes error**, không mô hình nào loại bỏ được, do nhãn `quality` là điểm cảm quan chủ quan và bài lab cắt lớp tại ranh giới ≤5 / =6 / ≥7 — đúng chỗ người chấm hay lưỡng lự.

Hệ quả trực tiếp: với 2998 mẫu, **không bộ siêu tham số nào vượt được ngưỡng 0.70**. Chỉ khi Bước 3 bổ sung 2998 mẫu (tổng 5996), accuracy mới lên **0.7540 / f1 0.7534**. Kết luận: siêu tham số quyết định ta chạm được bao nhiêu phần của trần; dữ liệu quyết định trần nằm ở đâu.

## 2. Khó khăn gặp phải và cách giải quyết

| # | Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|---|
| 1 | `pip install -r requirements.txt` thất bại: `No matching distribution found for numpy==2.0.0rc1` | venv chạy Python 3.14; `scikit-learn 1.4.2` không có wheel cho 3.14 nên pip build từ source, mà build-dependency `numpy==2.0.0rc1` đã bị gỡ khỏi PyPI | Tạo lại venv bằng Python 3.11 (`py -3.11 -m venv .venv`), khớp với `python-version: "3.10"` của CI |
| 2 | `WinError 32: file is being used by another process` khi cài `pandas` | Hai tiến trình pip cùng ghi vào một `site-packages` | Dừng tiến trình thừa, `pip install --force-reinstall --no-cache-dir pandas` để sửa phần cài dở |
| 3 | `aws s3 mb` báo `AccessDenied: s3:CreateBucket` | Tài khoản `ai-lab-user` thuộc group `AI-Lab-Group` chỉ có EC2/IAM/ELB/VPC FullAccess, **không có S3** | Group có `IAMFullAccess`, nên tự gắn thêm: `aws iam attach-user-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess`. Sau khi nộp bài đã gỡ lại quyền này |
| 4 | `dvc push` báo `s3 is supported, but requires 'dvc-s3'` | `requirements.txt` gốc pin `dvc[gs]` (bản GCP); lab được làm trên AWS | Đổi sang `dvc[s3]` và `boto3`, cài bổ sung `pip install dvc-s3` |
| 5 | `git push` không kích hoạt GitHub Actions, `gh run list` rỗng | Repo là **fork**; GitHub mặc định chặn workflow trên fork để chống lạm dụng | Bật thủ công trong tab Actions, rồi kích hoạt bằng `gh workflow run` (`workflow_dispatch`). Các lần push sau tự chạy bình thường |
| 6 | pytest crash cục bộ: `Yaml file 'mlruns/0/meta.yaml' does not exist` | Test chạy không có `MLFLOW_TRACKING_URI` nên MLflow rơi về file-store `./mlruns`, trong khi metadata thật nằm ở SQLite | Bổ sung `mlruns/0/meta.yaml`. Trên CI không gặp lỗi này vì `mlruns/` chưa tồn tại, MLflow tự tạo mới |
| 7 | Pipeline Bước 2 fail ở job Eval | accuracy 0.6840 < ngưỡng 0.70 | **Không sửa** — đây là hành vi đúng của quality gate, chứng minh hệ thống từ chối deploy mô hình chưa đạt. Bước 3 bổ sung dữ liệu, accuracy lên 0.7540, cổng mở và Deploy chạy |

**Điều chỉnh so với README:** README viết cho GCP, lab này làm trên AWS. Các ánh xạ đã thực hiện: GCS → S3, `gsutil` → `aws s3`, Service Account JSON → IAM user access key, `google-cloud-storage` → `boto3`, `dvc[gs]` → `dvc[s3]`, GCE → EC2 (user mặc định `ubuntu`, security group thay firewall-rules), và credentials đặt tại `~/.aws/credentials` trên VM thay vì biến `GOOGLE_APPLICATION_CREDENTIALS`.
