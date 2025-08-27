# pdf_to_s3_step2.py  (flow mới: skip nếu JSON đã ghi 'result.pdf': 'uploaded')
from __future__ import annotations
import os, re, json, unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import fitz  # PyMuPDF
import boto3
from botocore.exceptions import ClientError

# ====== Cấu hình LOCAL ======
INPUT_DIR    = Path("C:/Users/Nguyen Ngoc Phu/Downloads/20250822_Release_AI_REHASO_v1/data_raw/PDF/")              # thư mục chứa PDF
MAX_PAGES    = 1                             # số trang đầu để quét
STATE_JSON   = Path("C:/Users/Nguyen Ngoc Phu/Downloads/20250822_Release_AI_REHASO_v1/data_raw/PDF/upload_state_pdf.json")   # file JSON trạng thái

# ====== Cấu hình S3 ======
S3_BUCKET    = os.getenv("S3_BUCKET", "getting-started-with-s3-for-rehab")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
S3_PREFIX    = "RESULT"                      # gốc theo yêu cầu
SUBDIR       = "DIER"                        # nhánh con theo yêu cầu

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# ====== Regex ======
# Ưu tiên mẫu gộp:  Name: <Name> (* m/d/yyyy)
COMBINED_NAME_DOB = [
    r"(?:Name|Patient Name)\s*[:：]\s*([A-Za-zÀ-ỹ\s]+?)\s*\(\s*\*\s*(\d{1,2})[./-](\d{1,2})[./-](\d{4})\s*\)"
]

# Dự phòng: tách riêng Name / DOB theo nhiều biến thể
NAME_PATTERNS = [
    r"(?:Họ\s*[-\s]*tên|Họ\s* và\s* tên|Họ\s*tên|Tên bệnh nhân|Bệnh nhân|Patient Name|Full Name|Name)\s*[:：]\s*([A-Za-zÀ-ỹ\s]+)",
    r"(?:Patient)\s*[:：]\s*([A-Za-zÀ-ỹ\s]+)",
]
DOB_PATTERNS = [
    r"(?:Ngày\s*sinh|NS|Sinh ngày|DOB|Date of Birth)\s*[:：]\s*([0-9.\-/\s]+)",
    r"\b(\d{4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,2})\b",   # yyyy-mm-dd
    r"\b(\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{4})\b",   # dd-mm-yyyy
]

# ====== Helpers ======
def strip_accents(text: str) -> str:
    nf = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nf if not unicodedata.combining(c))

def reorder_name_first(name: str) -> str:
    """
    Đưa First name (từ cuối) lên đầu rồi PascalCase và bỏ khoảng trắng.
    Ví dụ:
        "Nguyen Van A" -> "ANguyenVan"
        "To Lam"       -> "LamTo"
    """
    cleaned = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    ascii_name = strip_accents(cleaned)
    parts = [p for p in ascii_name.split() if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].capitalize()
    reordered = [parts[-1].capitalize()] + [p.capitalize() for p in parts[:-1]]
    return "".join(reordered)

def normalize_dob(date_str: str) -> Optional[str]:
    """Chuẩn hoá về YYYYMMDD. Hỗ trợ dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd, có/không padding."""
    if not date_str:
        return None
    ds = re.sub(r"\s+", "", date_str.strip())

    # dd-mm-yyyy / dd.mm.yyyy / dd/mm/yyyy
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$", ds)
    if m:
        d, mth, y = m.groups()
        return f"{y}{int(mth):02d}{int(d):02d}"

    # yyyy-mm-dd / yyyy.mm.dd / yyyy/mm/dd
    m = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$", ds)
    if m:
        y, mth, d = m.groups()
        return f"{y}{int(mth):02d}{int(d):02d}"

    return None

def extract_text_first_pages(pdf_path: Path, max_pages: int = 3) -> str:
    try:
        doc = fitz.open(pdf_path)
        n = min(max_pages, len(doc))
        texts = []
        for i in range(n):
            page = doc.load_page(i)
            texts.append(page.get_text("text"))
        return "\n".join(texts)
    except Exception as e:
        print(f"[WARN] Không đọc được {pdf_path.name}: {e}")
        return ""

def find_first(patterns: List[str], text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            for g in reversed(m.groups()):
                if g:
                    return g.strip()
    return None

def parse_patient_info(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # 1) Thử mẫu gộp: Name + (* DOB)
    for pat in COMBINED_NAME_DOB:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            mm, dd, yy = m.group(2), m.group(3), m.group(4)
            dob_norm = f"{yy}{int(mm):02d}{int(dd):02d}"  # YYYYMMDD
            dob_raw = f"{mm}/{dd}/{yy}"
            return name, dob_raw, dob_norm

    # 2) Fallback: tách riêng Name và DOB
    name = find_first(NAME_PATTERNS, text)
    dob_raw = find_first(DOB_PATTERNS, text)
    dob = normalize_dob(dob_raw) if dob_raw else None
    return name, dob_raw, dob

def build_patient_id(dob_yyyymmdd: str, name: str) -> str:
    return f"{dob_yyyymmdd}_{reorder_name_first(name)}"

# ----- JSON state helpers -----
def load_state() -> Dict:
    if STATE_JSON.exists():
        try:
            return json.loads(STATE_JSON.read_text(encoding="utf-8"))
        except Exception:
            print("[WARN] upload_state.json hỏng hoặc rỗng, tạo mới.")
            return {}
    return {}

def save_state(state: Dict) -> None:
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# ----- S3 helpers -----
def s3_key_for(patient_id: Optional[str], basename: str) -> str:
    if patient_id:
        return f"{S3_PREFIX}/{patient_id}/{SUBDIR}/{basename}"
    else:
        return f"{S3_PREFIX}/_unmatched/{SUBDIR}/{basename}"

def upload_file_to_s3(local_path: Path, s3_key: str):
    try:
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        print(f"[OK] Uploaded -> s3://{S3_BUCKET}/{s3_key}")
    except ClientError as e:
        raise RuntimeError(f"S3 upload error: {e}")

def _unmatched_path(key: str) -> str:
    return f"s3://{S3_BUCKET}/{key}"

# ====== Core ======
def process_pdfs_and_upload():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    pdf_files = sorted([p for p in INPUT_DIR.glob("*.pdf") if p.is_file()])
    if not pdf_files:
        print(f"[NOTE] Không tìm thấy PDF trong {INPUT_DIR.resolve()}")
        return

    for pdf in pdf_files:
        try:
            text = extract_text_first_pages(pdf, max_pages=MAX_PAGES)
            name, dob_raw, dob = parse_patient_info(text)
            patient_id = build_patient_id(dob, name) if (name and dob) else None

            if not patient_id:
                # Không nhận diện được -> đẩy vào _unmatched, giữ tên gốc
                basename = pdf.name
                key = s3_key_for(None, basename)
                upload_file_to_s3(pdf, key)

                # Cập nhật state cho unmatched
                state.setdefault("_unmatched", {})
                state["_unmatched"][basename] = "uploaded_missing_id"
                print(f"[INFO] {pdf.name}: chưa nhận diện được ID_Patient → {_unmatched_path(key)}")
                continue

            # Có ID_Patient → dùng đúng 'result.pdf' theo quy ước
            per_patient = state.setdefault(patient_id, {})

            # === Flow mới: nếu đã có 'result.pdf': 'uploaded' -> SKIP ===
            if per_patient.get("result.pdf") == "uploaded":
                print(f"[SKIP] {pdf.name} → ID={patient_id} đã có result.pdf: uploaded trong JSON. Bỏ qua upload.")
                continue

            # Nếu chưa có hoặc chưa 'uploaded' → upload và ghi trạng thái
            basename = "result.pdf"
            key = s3_key_for(patient_id, basename)
            upload_file_to_s3(pdf, key)

            per_patient["result.pdf"] = "uploaded"
            print(f"[INFO] {pdf.name} → ID={patient_id} → {key}")

        except Exception as e:
            print(f"[ERR] {pdf.name}: {e}")

    save_state(state)
    print("\n[DONE] Hoàn tất. Đã cập nhật upload_state.json")

if __name__ == "__main__":
    process_pdfs_and_upload()
