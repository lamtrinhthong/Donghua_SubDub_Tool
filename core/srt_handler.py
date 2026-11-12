import re
from typing import List, Dict
from pathlib import Path
from core.common import seconds_to_srt_time, srt_time_to_seconds

__all__ = ["seconds_to_srt_time", "srt_time_to_seconds", "parse_srt", "align_tts_durations", "save_srt"]

def parse_srt(srt_path: Path) -> List[Dict]:
    """
    Đọc và phân tích file .srt thành danh sách các câu phụ đề.
    Hỗ trợ xử lý UTF-8 BOM và các dòng trống chứa khoảng trắng.
    """
    if not srt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file SRT: {srt_path}")

    # Dùng utf-8-sig để tự động loại bỏ ký tự BOM (\ufeff) ở đầu file
    with open(srt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")

    # Tách block kể cả khi dòng trống có chứa dấu cách/tab
    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles = []

    time_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            continue

        time_line_idx = -1
        match = None
        for i, line in enumerate(lines):
            match = time_pattern.search(line)
            if match:
                time_line_idx = i
                break

        if not match or time_line_idx == -1:
            continue

        try:
            sub_id = int(lines[0]) if time_line_idx > 0 else len(subtitles) + 1
        except ValueError:
            sub_id = len(subtitles) + 1

        start_sec = srt_time_to_seconds(match.group(1))
        end_sec = srt_time_to_seconds(match.group(2))
        text = "\n".join(lines[time_line_idx + 1:]).strip()

        subtitles.append({
            "id": sub_id,
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(max(0.001, end_sec - start_sec), 3),
            "text": text
        })

    return subtitles

def align_tts_durations(
    subtitles: List[Dict], 
    tts_durations: List[float], 
    max_speed: float = 1.35,
    min_gap: float = 0.05
) -> List[Dict]:
    """
    Hàm giải quyết lệch/trôi thời gian khi ghép TTS tiếng Việt vào timeline tiếng Trung:
    - Nếu audio tiếng Việt dài hơn khung giờ gốc, ưu tiên 'mượn' khoảng nghỉ (gap) 
      trước câu tiếp theo để giọng đọc tự nhiên nhất.
    - Nếu mượn hết gap mà vẫn dài hơn, tự động tính hệ số tăng tốc (speed_factor) 
      để dùng với ffmpeg atempo ép vừa khít khung giờ, tuyệt đối không đẩy lùi câu sau.
    """
    n = len(subtitles)
    aligned_subs = []

    for i, sub in enumerate(subtitles):
        item = sub.copy()
        actual_tts_dur = tts_durations[i] if i < len(tts_durations) else item["duration"]
        
        # Kiểm tra giới hạn tối đa có thể kéo dài câu hiện tại mà không đè lên câu sau
        if i < n - 1:
            next_start = subtitles[i + 1]["start"]
            max_allowed_end = max(item["end"], next_start - min_gap)
        else:
            max_allowed_end = item["start"] + actual_tts_dur

        max_available_window = round(max_allowed_end - item["start"], 3)

        if actual_tts_dur <= item["duration"]:
            # TTS ngắn hơn hoặc bằng sub gốc -> Giữ nguyên tốc độ 1.0x
            item["speed_factor"] = 1.0
            item["tts_end"] = round(item["start"] + actual_tts_dur, 3)
        elif actual_tts_dur <= max_available_window:
            # TTS dài hơn sub gốc nhưng vẫn nằm gọn trong khoảng nghỉ trước câu tiếp theo
            item["speed_factor"] = 1.0
            item["end"] = round(item["start"] + actual_tts_dur, 3)
            item["duration"] = round(actual_tts_dur, 3)
            item["tts_end"] = item["end"]
        else:
            # TTS tràn qua cả câu tiếp theo -> Tận dụng tối đa khoảng nghỉ + tăng tốc độ đọc
            item["end"] = round(max_allowed_end, 3)
            item["duration"] = max_available_window
            item["tts_end"] = item["end"]
            req_speed = actual_tts_dur / max_available_window
            item["speed_factor"] = round(min(req_speed, max_speed), 3)
            item["is_overflow"] = req_speed > max_speed  # Cảnh báo nếu câu dịch quá dài

        aligned_subs.append(item)

    return aligned_subs

def save_srt(subtitles: List[Dict], out_path: Path, text_key: str = "text") -> Path:
    """Xuất danh sách phụ đề ra file .srt chuẩn SubRip."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []

    for idx, sub in enumerate(subtitles, 1):
        start_str = seconds_to_srt_time(sub["start"])
        end_str = seconds_to_srt_time(sub["end"])
        txt = sub.get(text_key, "").strip()
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{txt}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")

    return out_path
