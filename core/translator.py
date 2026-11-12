import sys
import os
import json
import subprocess
import shutil
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.srt_handler import parse_srt, save_srt

# Cấu hình gọi Antigravity CLI (agy)
AGY_MODEL = "gemini-3.8-flash-low"
AGY_TIMEOUT = 300          # giây cho mỗi batch (agy có thể rất chậm với batch lớn)
AGY_BATCH_SIZE = 40       # số câu mỗi batch
AGY_MAX_WORKERS = 4       # số batch chạy song song (mỗi batch = 1 tiến trình agy)
AGY_CMD = ["agy", "--model", AGY_MODEL, "--output-format", "json", "--dangerously-skip-permissions"]

def translate_srt_file(
    input_srt_path: Path,
    output_srt_path: Path,
    model: str = AGY_MODEL,
    batch_size: int = AGY_BATCH_SIZE,
    context_hint: str = "",
    max_workers: int = AGY_MAX_WORKERS
) -> Path:
    """
    Dịch file .srt tiếng Trung sang file .srt tiếng Việt:
    - Bảo toàn 100% timecode [start --> end] của từng câu thoại.
    - Văn phong kiếm hiệp, tiên hiệp Hán-Việt tu tiên súc tích.
    - ĐỦ NGHĨA 100%: giữ nguyên danh xưng, tên riêng, con số, thần thú, pháp bảo.
    - Hỗ trợ context_hint từ Bilibili để chuẩn hóa tên nhân vật/môn phái theo ngữ cảnh.
    - Xuất ra file .srt tiếng Việt chuẩn. Người dùng có thể mở ra kiểm tra/sửa đổi tự do!
    """
    if not shutil.which("agy"):
        raise RuntimeError("Không tìm thấy lệnh 'agy' trong PATH! Vui lòng kiểm tra cài đặt Antigravity CLI.")

    subs = parse_srt(input_srt_path)
    if not subs:
        print("[!] File SRT rỗng!")
        return output_srt_path

    print(f"[*] [Translator] Đang dịch {len(subs)} câu từ {input_srt_path.name} sang tiếng Việt...")
    total_batches = (len(subs) + batch_size - 1) // batch_size
    trans_map = {}

    system_prompt = """Bạn là chuyên gia dịch thuật và biên kịch thuyết minh phim hoạt hình tu tiên / kiếm hiệp Trung Quốc sang tiếng Việt.
YÊU CẦU BẮT BUỘC ĐỂ KHỚP GIỌNG ĐỌC TỐC ĐỘ 1.0x:
1. Dịch TỪNG CÂU một, giữ nguyên định dạng: [ID] Lời dịch tiếng Việt.
2. Mỗi câu đều có chỉ dẫn (Tối đa X chữ). BẮT BUỘC dịch cô đọng bằng từ Hán-Việt chuẩn xác sao cho tổng số chữ của câu KHÔNG VƯỢT QUÁ X chữ.
   - Ví dụ: Nếu giới hạn (Tối đa 3 chữ), hãy dịch ngắn gọn như "Mau lui lại!", "Không thể nào!", "Chết đi!" thay vì viết dài dòng.
3. ĐỦ NGHĨA CỐT LÕI 100%: Giữ nguyên tên riêng, danh xưng (gia gia, phụ thân, sư tôn, đạo hữu), con số, pháp bảo, cảnh giới.
4. KHÔNG dùng dấu phẩy (,) không cần thiết trong các câu ngắn dưới 6 chữ để tránh giọng đọc bị ngắt nhịp lâu.
5. Chỉ xuất ra đúng định dạng: [ID] Lời dịch tiếng Việt (tuyệt đối không in lại phần ghi chú số chữ)."""

    if context_hint:
        system_prompt += f"\nNGỮ CẢNH PHIM & TỪ KHÓA TỪ BILIBILI: {context_hint} Hãy dịch thống nhất tên riêng theo ngữ cảnh này."

    # ------------------------------------------------------------------
    # Tách hàm gọi 1 batch (để chạy song song nhiều tiến trình agy)
    # ------------------------------------------------------------------
    def build_prompt(chunk: List[Dict], b_idx: int) -> str:
        prompt_lines = []
        for i_in_chunk, s in enumerate(chunk):
            if not s["text"].strip():
                continue
            global_idx = b_idx + i_in_chunk
            # Tính tổng thời gian khả dụng (bao gồm tối đa 0.6s khoảng nghỉ trước câu sau nếu có)
            if global_idx + 1 < len(subs):
                gap = max(0.0, subs[global_idx + 1]["start"] - s["end"])
                usable_sec = s["duration"] + min(0.6, gap * 0.8)
            else:
                usable_sec = s["duration"] + 0.5

            # Ở tốc độ 1.0x, giọng Việt đọc chuẩn 3.3 chữ/giây
            max_words = max(2, int(round(usable_sec * 3.3)))
            prompt_lines.append(f"[{s['id']}] (Tối đa {max_words} chữ) {s['text']}")
        if not prompt_lines:
            return ""
        input_payload = "\n".join(prompt_lines)
        return (f"### HƯỚNG DẪN:\n{system_prompt}\n\n"
                f"### DANH SÁCH CÂU THOẠI:\n{input_payload}\n\n"
                f"### BẢN DỊCH TIẾNG VIỆT:\n")

    def parse_response(resp_text: str) -> Dict[int, str]:
        result: Dict[int, str] = {}
        for line in resp_text.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                try:
                    idx = int(line[1:line.index("]")])
                    val = line[line.index("]") + 1:].strip()
                    val = re.sub(r'[\*\#\_\[\]\<\>\{\}]', '', val).strip()
                    val = re.sub(r'^\(?tối đa \d+ (?:từ|chữ)\)?[:\s-]*', '', val, flags=re.IGNORECASE).strip()
                    if val.endswith("..."):
                        val = val[:-3].strip()
                    if not val.endswith((".", "!", "?")):
                        val += "."
                    if val:
                        result[idx] = val
                except Exception:
                    pass
        return result

    def run_batch(b_idx: int) -> Dict[int, str]:
        """Gọi agy cho 1 batch, có retry. Trả về dict {id: text_vi}."""
        chunk = subs[b_idx:b_idx + batch_size]
        full_prompt = build_prompt(chunk, b_idx)
        if not full_prompt:
            return {}
        for attempt in range(3):
            try:
                p = subprocess.Popen(AGY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
                stdout_text, _ = p.communicate(input=full_prompt, timeout=AGY_TIMEOUT)
                if p.returncode == 0:
                    try:
                        data = json.loads(stdout_text.strip())
                        resp_text = data.get("response", "").strip()
                    except Exception:
                        resp_text = stdout_text.strip()
                    if resp_text:
                        return parse_response(resp_text)
                else:
                    print(f"[!] [Translator] agy trả về mã lỗi {p.returncode}, thử lại...")
            except subprocess.TimeoutExpired:
                print(f"[!] [Translator] agy timeout sau {AGY_TIMEOUT}s (batch {b_idx // batch_size + 1}), thử lại...")
                try:
                    p.kill()
                except Exception:
                    pass
            except Exception as e:
                print(f"[!] [Translator] Lỗi gọi agy: {e}")
            time.sleep(2 * (attempt + 1))
        return {}

    # ------------------------------------------------------------------
    # Chạy các batch SONG SONG (agy có overhead ~50s khởi động mỗi lần gọi)
    # ------------------------------------------------------------------
    batch_indices = list(range(0, len(subs), batch_size))
    total_batches = len(batch_indices)
    print(f"[*] [Translator] {len(subs)} câu / {total_batches} batch, chạy song song {max_workers} luồng...")

    results: Dict[int, Dict[int, str]] = {}
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_batch, b): b for b in batch_indices}
            done = 0
            for fut in as_completed(futures):
                b = futures[fut]
                try:
                    results[b] = fut.result()
                except Exception:
                    results[b] = {}
                done += 1
                print(f"    - Batch {b // batch_size + 1}/{total_batches} xong ({done}/{total_batches}, "
                      f"{sum(len(v) for v in results.values())} câu)")
    else:
        for b in batch_indices:
            results[b] = run_batch(b)
            print(f"    - Batch {b // batch_size + 1}/{total_batches} xong")

    for b in batch_indices:
        trans_map.update(results.get(b, {}))

    print(f"    - Đã dịch được {len(trans_map)}/{len(subs)} câu")

    # Kiểm tra các câu bị thiếu và retry tự động (chia nhỏ để tránh timeout)
    missing_items = [s for s in subs if s["id"] not in trans_map or re.search(r'[\u4e00-\u9fff]', trans_map[s["id"]])]
    if missing_items:
        print(f"[*] [Translator] Phát hiện {len(missing_items)} câu chưa dịch, tiến hành retry...")
        RETRY_CHUNK = 25
        for r_start in range(0, len(missing_items), RETRY_CHUNK):
            r_chunk = missing_items[r_start:r_start + RETRY_CHUNK]
            retry_lines = [f"[{s['id']}] {s['text']}" for s in r_chunk if s['text'].strip()]
            if not retry_lines:
                continue
            retry_prompt = f"### HƯỚNG DẪN:\n{system_prompt}\n\n### DANH SÁCH CÂU THOẠI CẦN DỊCH:\n" + "\n".join(retry_lines) + "\n\n### BẢN DỊCH TIẾNG VIỆT:\n"
            for attempt in range(2):
                try:
                    p = subprocess.Popen(AGY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
                    stdout_text, _ = p.communicate(input=retry_prompt, timeout=AGY_TIMEOUT)
                    if p.returncode == 0:
                        try:
                            data = json.loads(stdout_text.strip())
                            r_text = data.get("response", "").strip()
                        except Exception:
                            r_text = stdout_text.strip()
                        for line in r_text.splitlines():
                            line = line.strip()
                            if line.startswith("[") and "]" in line:
                                try:
                                    idx = int(line[1:line.index("]")])
                                    val = line[line.index("]") + 1:].strip()
                                    val = re.sub(r'[\*\#\_\[\]\<\>\{\}]', '', val).strip()
                                    if val: trans_map[idx] = val
                                except Exception:
                                    pass
                        break
                    print(f"[!] [Translator] Retry batch lỗi, thử lại...")
                except subprocess.TimeoutExpired:
                    print(f"[!] [Translator] Retry timeout sau {AGY_TIMEOUT}s")
                    try:
                        p.kill()
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[!] [Translator] Lỗi retry: {e}")
                time.sleep(2)

    translated_subs = []
    for s in subs:
        sid = s["id"]
        vi_text = trans_map.get(sid, "")
        if not vi_text or re.search(r'[\u4e00-\u9fff]', vi_text):
            print(f"[!] [Translator] Cảnh báo: Câu #{sid} '{s['text']}' không dịch được, giữ nguyên text gốc.")
            vi_text = s["text"]
        translated_subs.append({
            "id": sid,
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
            "text": vi_text
        })

    save_srt(translated_subs, output_srt_path)
    print(f"[OK] [Translator] Đã xuất bản dịch SRT tiếng Việt thành công: {output_srt_path.name}")
    return output_srt_path
