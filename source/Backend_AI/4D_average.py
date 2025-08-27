from __future__ import annotations

import os
import pandas as pd
from typing import Dict, Optional, Iterator, Tuple, List

import boto3
from botocore.exceptions import ClientError
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime, timedelta
import secrets
import textwrap
from pathlib import Path
import re
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# ====== Cấu hình ======
BASE_URL = "http://localhost:8000/r/"   # chạy local
SHORT_TTL_SEC = 3600                    # short link sống 1 giờ
S3_BUCKET = "getting-started-with-s3-for-rehab"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")

ROOT_RESULT = "RESULT/"
ROOT_DATA = "DATA/"
SUBDIR = "DIER"
FILENAME = "4D average.csv"

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Tham số sinh (tương đương Ollama options)
GENERATION_CONFIG = {
    "temperature": 0.2,
    "maxOutputTokens": 3000,   # gần với num_predict
    # "topP": 0.95,
    # "topK": 40,
}

# S3 client dùng chung
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION,
)

def get_csv_df_from_s3(patient_id: str) -> pd.DataFrame:
    """
    Đọc CSV trực tiếp từ S3 và trả về DataFrame.
    Giả định CSV chỉ có một hàng kết quả.
    """
    key = f"{ROOT_DATA}{patient_id}/{SUBDIR}/{FILENAME}"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey",):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy file: s3://{S3_BUCKET}/{key}")
        raise HTTPException(status_code=500, detail=f"Lỗi S3 khi đọc: {e}")

    # Đọc nội dung CSV trực tiếp từ stream
    df = pd.read_csv(obj["Body"])
    return df

# ================== GEMINI CALLS ==================
def call_gemini(prompt: str,
                model: Optional[str] = None,
                generation_config: Optional[dict] = None,
                timeout: int = 180) -> str:
    """
    Gọi Gemini REST API (generateContent) cho tác vụ text-only.
    Tài liệu: https://ai.google.dev/api/generate-content
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Thiếu GEMINI_API_KEY trong biến môi trường")

    mdl = model or GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": generation_config or GENERATION_CONFIG,
        # Có thể thêm "systemInstruction": {"parts": [{"text": "..."}]} nếu cần
    }

    try:
        r = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=timeout)
        # 429 / quota → nêu rõ để dễ debug
        if r.status_code == 429:
            raise HTTPException(status_code=429, detail=f"Quá hạn mức Gemini: {r.text}")
        r.raise_for_status()
        data = r.json()

        # Lấy text từ candidates
        candidates = data.get("candidates") or []
        if not candidates:
            raise HTTPException(status_code=502, detail=f"Gemini không trả về candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
        resp = "\n".join([t for t in texts if t])

        if not resp.strip():
            raise HTTPException(status_code=502, detail=f"Gemini trả về rỗng: {data}")
        return resp
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Lỗi gọi Gemini: {e}")


# ================== PROMPT TÓM TẮT ==================
SUMMARY_PROMPT_TEMPLATE = """
Bạn là trợ lý phục hồi chức năng. 
Hãy đọc bảng kết quả sau và viết báo cáo ngắn gọn, dễ hiểu cho bệnh nhân.

BẢNG KẾT QUẢ:
{table_text}

YÊU CẦU:
- Báo cáo tối đa 500 từ, văn phong thân thiện, dễ hiểu.
- Không chẩn đoán bệnh, chỉ mô tả xu hướng và mức độ (nhẹ / vừa / cần lưu ý).
- Chỉ chọn 3–5 điểm nổi bật (vượt chuẩn hoặc sát ngưỡng).
- Nếu hầu hết các chỉ số bình thường, hãy nhấn mạnh điều này để bệnh nhân yên tâm.
- Trả lời đúng theo bố cục sau:

NHẬN XÉT & KHUYẾN NGHỊ
----------------------
1. Tóm tắt ngắn:
   (Viết 2–3 câu tổng quan về kết quả, nêu rõ mức độ chung: bình thường, cần lưu ý nhẹ, hay cần theo dõi.)
-------------------------------------------
2. Các điểm cần chú ý:
   - Điểm 1 (nêu chỉ số và ý nghĩa ngắn gọn)
   - Điểm 2
   - Điểm 3
   (Có thể thêm đến 4–5 điểm, mỗi điểm 1–2 câu ngắn gọn.)
-------------------------------------------
3. Gợi ý cải thiện:
   - Gợi ý về tư thế (ngồi, đứng, sinh hoạt)
   - Gợi ý về bài tập (tên 1–2 bài tập cụ thể)
   - Gợi ý về theo dõi (khi nào cần kiểm tra lại, khi nào nên gặp chuyên gia)
""" 
#======================= TABLE SUMMARY =================================
# def normalize_spaces(s: str) -> str:
#     return re.sub(r"\s+", " ", s).strip()

# def extract_values(norm_text: str):
#     result = {sec: {k: None for k in TEMPLATE[sec]} for sec in TEMPLATE}
#     for section, fields in result.items():
#         for key in fields:
#             pat = PATTERNS.get(key)
#             if pat:
#                 m = re.search(pat, norm_text, flags=re.IGNORECASE)
#                 if m:
#                     fields[key] = m.group(1)
#     return result

# def render_section_txt(title: str, data: dict) -> str:
#     lines = [title,
#              "Parameter".ljust(34) + "Measured Value".ljust(18) + "Normative",
#              "-"*34 + "-"*18 + "-"*20]
#     for key, val in data.items():
#         mv = (str(val) if val is not None else "")
#         norm = NORMATIVE.get(key, "")
#         lines.append(key.ljust(34) + mv.ljust(18) + norm)
#     lines.append("")
#     return "\n".join(lines)

# def write_summary_txt(result: dict, out_path: Path):
#     parts = []
#     for section in ["Spinal", "Pelvis", "Body"]:
#         parts.append(render_section_txt(section, result[section]))
#     out_path.write_text("\n".join(parts), encoding="utf-8")
    
def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    
#==================== FORMAT TABLE ===========================
# ===== 1) Bảng normative bạn đã có =====
NORMATIVE_LOOKUP = {
    # Spinal
    "Vertebral Rotation (RMS)": "0–3°",
    "Vertebral Rotation (+max)": "0–3°",
    "Vertebral Rotation (-max)": "0–3°",
    "Apical Deviation (VP-DM) (RMS)": "0–5 mm",
    "Apical Deviation (+max)": "0–5 mm",
    "Apical Deviation (-max)": "0–5 mm",
    "Sagittal Imbalance": "≤ 5°",
    "Coronal Imbalance": "≤ 5 mm",
    "Kyphotic Angle (ICT-ITL max)": "25–45°",
    "Lordotic Angle (ITL-ILS max)": "50–70°",
    "Cervical Flèche": "30–70 mm",
    "Lombaire Flèche": "30–70 mm",
    # Pelvis
    "Pelvic Obliquity": "≤ 1 mm",
    "Pelvic Torsion": "≤ 2°",
    "Pelvis Rotation": "≤ 5°",
    # Body
    "Trunk Length (VP-DM)": "380–460 mm",
    "Dimple Distance (DL-DR)": "70–110 mm",
}

# ===== 2) Khai báo nhóm + mapping từ tên hiển thị -> cột trong df -> khóa tra normative =====
# Mỗi item: (display_label, df_key, normative_key)
GROUPS = {
    "Thông tin cá nhân": [
        ("Female/Male",                  "Female/Male",                           None),
        ("Age",                          "Age",                                   None),
        ("Body height",                  "Body height",                           None),
        ("Body weight",                  "Body weight",                           None),
    ],
    "Cân bằng cơ thể": [
        ("Sagittal Imbalance [°]",       "Sagittal Imbalance VP-DM [°]",          "Sagittal Imbalance"),
        ("Sagittal Imbalance [mm]",      "Sagittal Imbalance VP-DM [mm]",         None),  # bạn đang dùng chuẩn theo °, mm thì để trống
        ("Coronal Imbalance [°]",        "Coronal Imbalance VP-DM [°]",           None),  # chuẩn thường theo mm -> để trống
        ("Coronal Imbalance [mm]",       "Coronal Imbalance VP-DM [mm]",          "Coronal Imbalance"),
    ],
    "Xương chậu": [
        ("Pelvic Obliquity [°]",         "Pelvic Obliquity [°]",                  None),  # chuẩn theo mm
        ("Pelvic Obliquity [mm]",        "Pelvic Obliquity [mm]",                 "Pelvic Obliquity"),
        ("Pelvic Torsion DL-DR [°]",     "Pelvic Torsion DL-DR [°]",              "Pelvic Torsion"),
        ("Pelvic Inclination (Dimples) [°]","Pelvic Inclination (Dimples) [°]",   None),
        ("Pelvis Rotation [°]",          "Pelvis Rotation [°]",                   "Pelvis Rotation"),
    ],
    "Độ cong cột sống": [
        ("Kyphotic Angle VP-ITL [°]",    "Kyphotic Angle VP-ITL [°]",             "Kyphotic Angle (ICT-ITL max)"),
        ("Kyphotic Angle VP-T12 [°]",    "Kyphotic Angle VP-T12 [°]",             "Kyphotic Angle (ICT-ITL max)"),  # gần nghĩa
        ("Lordotic Angle ITL-ILS (max) [°]","Lordotic Angle ITL-ILS (max) [°]",   "Lordotic Angle (ITL-ILS max)"),
        ("Lordotic Angle T12-DM [°]",    "Lordotic Angle T12-DM [°]",             None),
    ],
    "Vẹo cột sống": [
        ("Vertebral Rotation (rms) [°]", "Vertebral Rotation (rms) [°]",          "Vertebral Rotation (RMS)"),
        ("Vertebral Rotation (max) [°]", "Vertebral Rotation (max) [°]",          "Vertebral Rotation (+max)"),
        ("Apical Deviation (rms) [mm]",  "Apical Deviation VP-DM (rms) [mm]",     "Apical Deviation (VP-DM) (RMS)"),
        ("Apical Deviation (max) [mm]",  "Apical Deviation VP-DM (max) [mm]",     "Apical Deviation (+max)"),
    ],
    "Vai & thân": [
        ("Shoulder Obliquity [deg]",     "Shoulder Obliquity [deg]",              None),
        ("Shoulder Obliquity [mm]",      "Shoulder Obliquity [mm]",               None),
        ("Trunk Inclination ICT-ITL [deg]","Trunk Inclination ICT-ITL [deg]",     None),
    ]
}

# ===== 3) Hàm formatter 3 cột: Parameter | Measured | Normative =====
def format_grouped_block_with_normative(df, groups, normative_lookup, title="1) BẢNG CHỈ SỐ:"):
    """
    Xuất plain text có tiêu đề nhóm và 3 cột (Parameter, Measured, Normative).
    groups: dict[str, list[tuple(display_label, df_key, normative_key)]]
    """
    import math
    row = df.iloc[0].to_dict()

    def is_blank(v):
        return v is None or v == "" or (isinstance(v, float) and math.isnan(v))

    def val(v):
        return "—" if is_blank(v) else str(v)

    # Gom tất cả để tính độ rộng cột
    all_params   = []
    all_measured = []
    all_norms    = []
    for section, items in groups.items():
        for (label, df_key, norm_key) in items:
            all_params.append(label)
            all_measured.append(val(row.get(df_key)))
            all_norms.append(normative_lookup.get(norm_key, "—") if norm_key else "—")

    w_param = max(len("Parameter"), max(len(s) for s in all_params)) if all_params else len("Parameter")
    w_meas  = max(len("Measured"),  max(len(s) for s in all_measured)) if all_measured else len("Measured")
    w_norm  = max(len("Normative"), max(len(s) for s in all_norms)) if all_norms else len("Normative")

    # Helper render 1 hàng
    def line(param, measured, norm):
        return f"{param:<{w_param}} : {measured:<{w_meas}} : {norm}"

    lines = ["-" * len(title)]
    header = line("Parameter", "Measured", "Normative")
    sep    = "-" * len(header)
    for section, items in groups.items():
        lines += ["", section, "-" * len(section), header, sep]
        for (label, df_key, norm_key) in items:
            measured = val(row.get(df_key))
            normative = normative_lookup.get(norm_key, "—") if norm_key else "—"
            lines.append(line(label, measured, normative))
    return "\n".join(lines)

#===================== Check summary existed ===================
def exists_in_s3(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        # 404 / NoSuchKey / NotFound => chưa tồn tại
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        # Các lỗi khác thì ném ra để không âm thầm bỏ qua
        raise

#==================== Upload file to clound ====================
def upload_file_to_s3(local_file: Path, bucket: str, key: str):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(str(local_file), bucket, key)
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        print(f"✅ Upload thành công: {url}")
        return url
    except ClientError as e:
        print(f"❌ Lỗi upload: {e}")
        return None
    
# ================== TÓM TẮT KẾT QUẢ BẰNG GEMINI ==================
def tool_summarize_result(patient_id: str | int,
                          model: Optional[str] = None,
                          generation_config: Optional[dict] = None) -> Dict:
    """
    - Kiểm tra xem summary đã có chưa
    - Nếu có --> trả URL để API gọi lên cho UI
    - Nếu chưa --> gọi Gemini và trả kết quả
    """
    
    out_key = f"{ROOT_DATA}{patient_id}/{SUBDIR}/summary.txt"

    if exists_in_s3(S3_BUCKET, out_key):
        print("case1")
        url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": out_key},
        ExpiresIn=3600,  # 1 giờ
        )
        print("case1-end")
        return {"summary": None, "summary_link": url}

    print("case2-begin")
    df = get_csv_df_from_s3(patient_id)
    row = df.iloc[0]
    table = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    text =  format_grouped_block_with_normative(df, GROUPS, NORMATIVE_LOOKUP)
        
    prompt = SUMMARY_PROMPT_TEMPLATE.format(table_text=table)
        
    print(prompt)
    summary = call_gemini(prompt, model=model, generation_config=generation_config or GENERATION_CONFIG)

    OUT_REPORT_PATH = Path("table_report_csv.txt")   # bản tổng hợp cuối
    report = (
    f"BÁO CÁO TỔNG HỢP CỘT SỐNG & TƯ THẾ\n"
    f"{'-'*40}\n\n"
    f"1) BẢNG CHỈ SỐ:\n{text}\n\n"
    f"2) NHẬN XÉT & KHUYẾN NGHỊ:\n{summary}\n"
    )
    write_text(OUT_REPORT_PATH, report)
    upload_file_to_s3(OUT_REPORT_PATH, S3_BUCKET, out_key )
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": out_key},
        ExpiresIn=3600,  # 1 giờ
        )
    return {"summary": summary.strip(), "summary_link": url}

# ====== FastAPI ======
app = FastAPI(title="Patient Lookup API")

# CORS (allow your ngrok/localhost UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # DEV: cho tất cả origin, gồm cả 'null'
    allow_methods=["*"],       # GET, POST, OPTIONS,...
    allow_headers=["*"],       # Content-Type, Authorization,...
    allow_credentials=False,   # để False nếu không dùng cookie/Authorization
)

    
class LookupIn(BaseModel):
    id: str
    # tuỳ chọn: yêu cầu tạo tóm tắt bằng Gemini, upload và trả thêm link
    want_summary: Optional[bool] = False


def external_short_url(request: Request, key: str) -> str:
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    host   = request.headers.get("x-forwarded-host")  or request.headers.get("host")
    return f"{scheme}://{host}/r/{key}"


# key -> (full_url, expires_utc)
_short_store: Dict[str, Tuple[str, datetime]] = {}

def _put_short(full_url: str, ttl_sec: int = SHORT_TTL_SEC) -> str:
    for length in (6, 7, 8):
        key = secrets.token_urlsafe(length)[:length]
        if key not in _short_store:
            break
    _short_store[key] = (full_url, datetime.utcnow() + timedelta(seconds=ttl_sec))
    return key  # trả về key

def _get_short(key: str) -> Optional[str]:
    """Lấy URL từ key, kiểm tra hạn, tự xoá nếu hết hạn."""
    rec = _short_store.get(key)
    if not rec:
        return None
    url, exp = rec
    if datetime.utcnow() > exp:
        _short_store.pop(key, None)
        return None
    return url

def s3_key_for_patient(pid: str) -> str:
    return f"{ROOT_RESULT}{pid.strip()}/{SUBDIR}/result.pdf"

def ensure_exists_and_presign(bucket: str, key: str, expires: int = 3600) -> str:
    """Kiểm tra file tồn tại và trả presigned GET URL."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy file: s3://{bucket}/{key}")
        raise HTTPException(status_code=500, detail=f"Lỗi S3 khi kiểm tra: {e}")

    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
        return url
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo presigned URL: {e}")
    
    
# ====== ADD: resolver/redirect endpoint ======
@app.get("/r/{key}")
def resolve_short(key: str):
    url = _get_short(key)
    if not url:
        raise HTTPException(status_code=404, detail="Link expired or not found")
    # Dùng 303 để browser theo Location tới presigned URL
    return Response(status_code=303, headers={"Location": url})

# ---- POST /lookup: trả link file kết quả (result.pdf) ----
# ====== MODIFY: /lookup để trả short_url ======

@app.post("/lookup")
def lookup(body: LookupIn, request: Request) -> Dict:
    pid = (body.id or "").strip()
    print("ID patient is",pid)
    if not pid:
        raise HTTPException(status_code=400, detail="Thiếu ID bệnh nhân")

    key = s3_key_for_patient(pid)
    print("The key is created:",key)
    result_url = ensure_exists_and_presign(S3_BUCKET, key)
    print("result url is",result_url)

    res = tool_summarize_result(pid)
    summary_url = res.get("summary_link")
    print("summary url is:", summary_url)

    # create short keys (your existing _put_short/_get_short)
    short_key = _put_short(result_url)
    short_url = external_short_url(request, short_key)

    summary_short_url = None
    if summary_url:
        sum_key = _put_short(summary_url)
        summary_short_url = external_short_url(request, sum_key)

    return {
        "patient_id": pid,
        "key": key,
        "result_url": result_url,
        "summary_url": summary_url,
        "short_url": short_url,
        "summary_short_url": summary_short_url,
        "short_ttl_seconds": SHORT_TTL_SEC,
    }

    
# # ===== Main =====
# def main():
#         pid = f"19720912_TrungNguyenTan"
#         res = tool_summarize_result(pid)  # thay patient_id thực tế
#         print(res["summary_link"])
#         # print("heelooooooo")
# if __name__ == "__main__":
#     main()
    