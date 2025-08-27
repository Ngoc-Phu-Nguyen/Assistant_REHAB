import pandas as pd
from datetime import datetime
import os
import re
import json
from pathlib import Path
import boto3

    
# ==== Config ====
DATA_DIR = Path("C:/Users/Nguyen Ngoc Phu/Downloads/20250822_Release_AI_REHASO_v1/DATA/")
STATE_FILE = DATA_DIR / "upload_state.json"
BUCKET     = "getting-started-with-s3-for-rehab"
AWS_REGION = "ap-southeast-1"

s3 = boto3.client("s3", region_name=AWS_REGION)


def make_patient_id(row):
    # Birthday format YYYYMMDD
    try:
        bday = pd.to_datetime(row["Birthday"], errors="coerce")
        bday_str = bday.strftime("%Y%m%d") if pd.notnull(bday) else "00000000"
    except Exception:
        bday_str = "00000000"
    
    # Full name ghép lại
    name_parts = [str(row.get("Last name", "")), 
                  str(row.get("First name", "")), 
                  str(row.get("Middle name", ""))]
    name = " ".join([p for p in name_parts if p and p != "nan"]).strip()
    
    # Ghép full name, bỏ khoảng trắng và ký tự đặc biệt
    clean_name = re.sub(r"[^A-Za-z0-9]", "", name)
    return f"{bday_str}_{clean_name}"



# ===== Bước 1: Đọc file đã clean =====
file_path = "C:/Users/Nguyen Ngoc Phu/Downloads/20250822_Release_AI_REHASO_v1/data_raw/4D average/DataExport.ini"
df = pd.read_csv(file_path, sep="\t", encoding="utf-16")

#====== Remove row x2 for one patient =======
exam_col = "Examination index" if "Examination index" in df.columns else None
# convert an toàn sang số
df["_EXAM_NUM"] = pd.to_numeric(df[exam_col], errors="coerce")
df1 = df[df["_EXAM_NUM"] == 1].copy()


# ===== Bước 2: Đổi "-" -> NaN =====
df1 = df1.replace("-", pd.NA)

# ===== Bước 3: Loại bỏ cột toàn bộ trống (NaN) =====
df_clean = df1.dropna(axis=1, how="all")
print("Sau khi loại bỏ cột trống hoặc toàn '-':", df_clean.shape)

# ===== Bước 4: Nếu có cột Birthday và Created thì tính tuổi =====
df_clean = df_clean.copy()
if "Birthday" in df_clean.columns and "Created" in df_clean.columns:
    # parse datetime
    df_clean["Birthday"] = pd.to_datetime(df_clean["Birthday"], errors="coerce")
    df_clean["Created"] = pd.to_datetime(df_clean["Created"], errors="coerce", dayfirst=True)
    
    # Tính tuổi tại thời điểm đo (Created - Birthday)
    df_clean["Age"] = (df_clean["Created"] - df_clean["Birthday"]).dt.days // 365

# # ===== Bước 5: Bỏ các cột không cần thiết =====
# Danh sách cột quan trọng cần giữ lại
keep_cols = [
    # Thông tin cơ bản
    "Female/Male", "Age", "Body height", "Body weight",
    
    # Cân bằng cơ thể
    "Sagittal Imbalance VP-DM [°]", "Sagittal Imbalance VP-DM [mm]",
    "Coronal Imbalance VP-DM [°]", "Coronal Imbalance VP-DM [mm]",
    
    # Xương chậu
    "Pelvic Obliquity [°]", "Pelvic Obliquity [mm]",
    "Pelvic Torsion DL-DR [°]", "Pelvic Inclination (Dimples) [°]",
    "Pelvis Rotation [°]",
    
    # Độ cong cột sống
    "Kyphotic Angle VP-ITL [°]", "Kyphotic Angle VP-T12 [°]",
    "Lordotic Angle ITL-ILS (max) [°]", "Lordotic Angle T12-DM [°]",
    
    # Vẹo cột sống
    "Vertebral Rotation (rms) [°]", "Vertebral Rotation (max) [°]",
    "Apical Deviation VP-DM (rms) [mm]", "Apical Deviation VP-DM (max) [mm]",
    
    # Vai & thân
    "Shoulder Obliquity [deg]", "Shoulder Obliquity [mm]",
    "Trunk Inclination ICT-ITL [deg]"
]

# Lọc theo danh sách
cols_to_keep = [c for c in keep_cols if c in df_clean.columns]
df_summary = df_clean[cols_to_keep]
df_summary = df_summary.copy()

def sanitize(s: str) -> str:
    # bỏ ký tự không hợp lệ trong tên thư mục/file (Windows-safe)
    s = str(s)
    return re.sub(r'[\\/:*?"<>|]', "_", s)

root_dir = "C:/Users/Nguyen Ngoc Phu/Downloads/20250822_Release_AI_REHASO_v1/DATA/"
os.makedirs(root_dir, exist_ok=True)

df_summary["PatientID"] = df_clean.apply(make_patient_id, axis=1)
filename = f"4D average.csv"

for pid, group in df_summary.groupby("PatientID", dropna=False):
    safe_pid = sanitize(pid)

 
    patient_dir = os.path.join(root_dir, safe_pid,"DIER")
    os.makedirs(patient_dir, exist_ok=True)

    out_path = os.path.join(patient_dir, filename)
    group.drop("PatientID", axis=1, inplace=True)
    group.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Đã lưu {out_path}")
    

# ==== Load state ====
if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
else:
    state = {}
filename = f"4D average.csv"
def upload_to_s3(local_file: Path, patient_id: str):
    key = f"DATA/{patient_id}/DIER/{filename}"
    print(f"↑ Upload {local_file} -> s3://{BUCKET}/{key}")
    s3.upload_file(str(local_file), BUCKET, key)
    return "uploaded"

# ==== Loop over patients ====
for patient_dir in DATA_DIR.iterdir():
    if not patient_dir.is_dir():
        continue

    patient_id = patient_dir.name
    patient_state = state.get(patient_id, {})

    for root, _, files in os.walk(patient_dir):
        for file in files:
            local_file = Path(root) / file
            # đường dẫn tương đối (so với folder bệnh nhân)
            rel_path = str(local_file.relative_to(patient_dir)).replace("\\", "/")

            # Nếu file chưa có trong state thì upload
            if rel_path not in patient_state:
                status = upload_to_s3(local_file, patient_id)
                patient_state[rel_path] = status

    # cập nhật state cho bệnh nhân này
    state[patient_id] = patient_state

# ==== Save state ====
STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
print("✅ Upload hoàn tất, đã cập nhật state.")
