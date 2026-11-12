import sys
import subprocess
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import edge_tts
from PIL import ImageFont

from core.common import FFMPEG, format_ass_time
from core.srt_handler import parse_srt

from core.micro_pid_time_warper import MicroPIDTimeWarper

def wrap_text_vi(text: str, font, max_w: int = 680) -> Tuple[str, int, int]:
    words = text.split()
    if not words:
        return "", 0, 1
    lines = []
    curr = []
    for w in words:
        cand = " ".join(curr + [w])
        bbox = font.getbbox(cand)
        if (bbox[2] - bbox[0]) <= max_w:
            curr.append(w)
        else:
            if curr:
                lines.append(" ".join(curr))
                curr = [w]
            else:
                lines.append(w)
                curr = []
    if curr:
        lines.append(" ".join(curr))
    if len(lines) == 1:
        w = font.getbbox(lines[0])[2] - font.getbbox(lines[0])[0]
        return lines[0], w, 1
    elif len(lines) == 2:
        w1 = font.getbbox(lines[0])[2] - font.getbbox(lines[0])[0]
        w2 = font.getbbox(lines[1])[2] - font.getbbox(lines[1])[0]
        return f"{lines[0]}\\N{lines[1]}", max(w1, w2), 2
    else:
        mid = len(words) // 2
        l1, l2 = " ".join(words[:mid]), " ".join(words[mid:])
        w1 = font.getbbox(l1)[2] - font.getbbox(l1)[0]
        w2 = font.getbbox(l2)[2] - font.getbbox(l2)[0]
        return f"{l1}\\N{l2}", max(w1, w2), 2

def fetch_google_tts(text: str, out_mp3: Path):
    if out_mp3.exists() and out_mp3.stat().st_size > 500:
        return True
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]
    words = text.split()
    chunks = []
    curr = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + 1 > 180:
            chunks.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += len(w) + 1
    if curr:
        chunks.append(" ".join(curr))

    full_audio = bytearray()
    for chunk in chunks:
        success = False
        for attempt in range(4):
            ua = user_agents[attempt % len(user_agents)]
            headers = {"User-Agent": ua, "Referer": "https://translate.google.com/"}
            url = "https://translate.google.com/translate_tts"
            params = {"ie": "UTF-8", "q": chunk, "tl": "vi", "client": "tw-ob"}
            try:
                r = requests.get(url, params=params, headers=headers, timeout=12)
                if r.status_code == 200 and len(r.content) > 100:
                    full_audio.extend(r.content)
                    success = True
                    break
                elif r.status_code == 429:
                    time.sleep(2.5 * (attempt + 1))
            except Exception:
                time.sleep(1.0)
        if not success:
            from gtts import gTTS
            tts = gTTS(text=chunk, lang="vi")
            tmp = out_mp3.with_suffix(".tmp.mp3")
            tts.save(str(tmp))
            with open(tmp, "rb") as f:
                full_audio.extend(f.read())
            tmp.unlink(missing_ok=True)

    with open(out_mp3, "wb") as f:
        f.write(full_audio)
    return True


# Danh sách các preset giọng đọc để hiển thị lên UI / CLI của Tool
VOICE_PRESETS = {
    "nam_minh": {
        "engine": "edge",
        "voice": "vi-VN-NamMinhNeural",
        "rate": "+0%",
        "pitch": "-2Hz",  # Hạ nhẹ 2Hz giúp giọng nam trầm uy lực, đậm chất tiên hiệp/kiếm hiệp
        "desc": "Nam Minh (Microsoft Neural - Trầm ấm, chuẩn thuyết minh tu tiên)"
    },
    "nam_minh_tram": {
        "engine": "edge",
        "voice": "vi-VN-NamMinhNeural",
        "rate": "+0%",
        "pitch": "-6Hz",  # Giọng nam trầm sâu đặc biệt
        "desc": "Nam Minh Trầm Sâu (Giọng dày, hợp phim mạt thế/hành động)"
    },
    "hoai_my": {
        "engine": "edge",
        "voice": "vi-VN-HoaiMyNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "desc": "Hoài My (Microsoft Neural - Nữ truyền cảm, tự nhiên)"
    },
    "hoai_my_tre": {
        "engine": "edge",
        "voice": "vi-VN-HoaiMyNeural",
        "rate": "+0%",
        "pitch": "+4Hz",  # Giọng nữ cao, trẻ trung
        "desc": "Hoài My Trẻ Trung (Giọng trong sáng, hợp hoạt hình vui tươi/hài hước)"
    },
    "google_female": {
        "engine": "google",
        "voice": "vi",
        "rate": "+0%",
        "pitch": "+0Hz",
        "desc": "Chị Google (Mặc định - Nhanh, ổn định)"
    }
}


async def _edge_tts_async(text: str, out_mp3: Path, voice: str, rate: str = "+0%", pitch: str = "+0Hz"):
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(out_mp3))


def synthesize_tts_audio(text: str, out_mp3: Path, voice_id: str = "nam_minh") -> bool:
    """
    Bộ tổng hợp giọng nói đa luồng hỗ trợ chọn giọng (Nam Minh, Hoài My, Google TTS)
    kèm cơ chế tự động Fallback sang Google TTS nếu mạng chập chờn.
    """
    if out_mp3.exists() and out_mp3.stat().st_size > 500:
        return True

    preset = VOICE_PRESETS.get(voice_id, VOICE_PRESETS["nam_minh"])

    # 1. Nếu chọn nhánh Microsoft Edge Neural TTS (Nam Minh / Hoài My)
    if preset["engine"] == "edge":
        for attempt in range(3):
            try:
                asyncio.run(
                    _edge_tts_async(
                        text=text,
                        out_mp3=out_mp3,
                        voice=preset["voice"],
                        rate=preset.get("rate", "+0%"),
                        pitch=preset.get("pitch", "+0Hz")
                    )
                )
                if out_mp3.exists() and out_mp3.stat().st_size > 500:
                    return True
            except Exception:
                time.sleep(0.8 * (attempt + 1))
        print(f"[!] [TTS] Edge-TTS quá tải ở câu '{text[:20]}...', tự động fallback sang Google TTS.")

    # 2. Nhánh Google TTS (hoặc Fallback khi Edge-TTS lỗi)
    return fetch_google_tts(text, out_mp3)


def dub_from_srt(
    input_srt_path: Path,
    out_audio_path: Path,
    out_ass_path: Path,
    total_duration: float,
    voice_id: str = "nam_minh",  # <-- Thêm option chọn giọng ("nam_minh", "nam_minh_tram", "hoai_my", "hoai_my_tre", "google_female")
    work_dir: Optional[Path] = None,
    threads: int = 8,
    voice: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Tạo track âm thanh thuyết minh và phụ đề Dynamic Rounded Capsule từ file .srt:
    - Hỗ trợ lựa chọn giọng đọc Edge Neural TTS (Nam Minh / Hoài My) hoặc Google TTS.
    - Micro-PID Time Warper đảm bảo sai số trôi dạt = 0.00ms.
    - Nền đen Dynamic Capsule CHỈ ôm khít text tiếng Việt.
    """
    if voice is not None:
        voice_id = voice

    work_dir = work_dir or out_audio_path.parent / "temp_dub"
    # Tách riêng cache cho từng giọng để đổi giọng là nhận file mới ngay
    tts_dir = work_dir / f"tts_cache_{voice_id}"
    tts_dir.mkdir(parents=True, exist_ok=True)

    preset_info = VOICE_PRESETS.get(voice_id, VOICE_PRESETS["nam_minh"])
    subs = parse_srt(input_srt_path)
    print(f"[*] [Dubber] Đang tổng hợp {len(subs)} câu bằng giọng [{preset_info['desc']}]...")

    diff_input = [
        {
            "id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
            "text_vi": s["text"],
            "text_zh": ""
        }
        for s in subs
    ]

    def synthesize_single(item):
        sid = item["id"]
        text = item["text_vi"]
        mp3_file = tts_dir / f"sent_{sid:04d}.mp3"
        wav_file = tts_dir / f"sent_{sid:04d}.wav"
        if not (wav_file.exists() and wav_file.stat().st_size > 500):
            synthesize_tts_audio(text, mp3_file, voice_id=voice_id)
            subprocess.run([
                FFMPEG, "-nostdin", "-y",
                "-i", str(mp3_file),
                "-ar", "48000", "-ac", "1",
                str(wav_file)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return sid

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(synthesize_single, item) for item in diff_input]
        done = 0
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 50 == 0 or done == len(diff_input):
                print(f"    - TTS xong {done}/{len(diff_input)} câu ({(time.time()-t0):.1f}s)")

    print(f"[OK] [Dubber] Hoàn tất toàn bộ {len(diff_input)} file audio câu thoại.")

    # Chạy trực tiếp Micro-PID Time Warper trên timeline gốc từ SRT
    print(f"[*] [Dubber] Đồng bộ dòng thời gian bằng Micro-PID Time Warper (Tốc độ: 1.00x - 1.05x, cứu hộ tới 1.40x)...")
    warper = MicroPIDTimeWarper(
        sample_rate=48000,
        min_speed=1.00,
        max_speed=1.05,
        emergency_speed=1.20,
        hard_cap_speed=1.40,
        min_breath_pause=0.025
    )
    diff_events = warper.differentiate_timeline(diff_input, total_duration=total_duration)
    out_audio_path, scheduled_events, metrics = warper.process_and_synthesize(
        diff_events=diff_events,
        tts_cache_dir=tts_dir,
        total_duration=total_duration,
        out_audio_path=out_audio_path
    )
    print(f"[OK] [Dubber] Master Audio sẵn sàng: {out_audio_path.name} (Tốc độ TB: {metrics['mean_speed']}x, Drift TB: {metrics['mean_drift_ms']}ms)")
    if metrics.get("trimmed_sentences"):
        print(f"[!] [Dubber] {metrics['trimmed_sentences']} câu bị cắt cứng để tránh chồng lấn "
              f"(tổng {metrics['total_trimmed_ms']}ms) - có thể tăng hard_cap_speed để giữ đủ lời.")
    if metrics.get("delayed_sentences"):
        print(f"[!] [Dubber] {metrics['delayed_sentences']} câu bị trễ khỏi timeline gốc "
              f"(tối đa {metrics['max_overlap_ms']}ms) - nghĩa là lời dịch dài hơn khung thời gian.")
    if metrics.get("max_speed_used", 1.0) > 1.20:
        print(f"[!] [Dubber] Có câu chạy tối đa {metrics['max_speed_used']}x - nghe có thể hơi gấp.")

    # Sinh file ASS Dynamic Rounded Capsule ôm khít text tiếng Việt
    print(f"[*] [Dubber] Đang sinh file phụ đề Dynamic Rounded Capsule ASS: {out_ass_path.name}...")
    try:
        font_arial = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 25)
    except Exception:
        font_arial = ImageFont.load_default()

    ass_header = """[Script Info]
Title: FastLab Dynamic Rounded Capsule (Vietnamese Only)
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BgBox,Arial,20,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: SubText,Arial,25,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    r = 16
    k = int(r * 0.55228475)

    for ev in scheduled_events:
        text_vi = ev.get("text_vi", "").strip()
        if not text_vi:
            continue
        s_val = ev.get("actual_start", ev.get("start", 0.0))
        # Dùng `sub_end` (thời điểm hiển thị đã tính sẵn) thay vì `actual_end`,
        # để phụ đề khớp chính xác với audio kể cả khi câu bị cắt cứng.
        e_val = ev.get("sub_end", ev.get("actual_end", ev.get("end", s_val + 2.0)))
        e_val = max(e_val, s_val + 0.05)
        s_str = format_ass_time(s_val)
        e_str = format_ass_time(e_val)

        wrapped_vi, w_vi, num_lines = wrap_text_vi(text_vi, font_arial, max_w=680)
        box_w = max(160, min(750, w_vi + 36))  # Ôm sát text tiếng Việt
        box_h = 76 if num_lines == 2 else 50
        cy = 642

        hw = box_w // 2
        cx = 640
        x1, x2 = cx - hw, cx + hw
        y1, y2 = cy - box_h // 2, cy + box_h // 2

        path = (
            f"m {x1+r} {y1} "
            f"l {x2-r} {y1} "
            f"b {x2-r+k} {y1} {x2} {y1+r-k} {x2} {y1+r} "
            f"l {x2} {y2-r} "
            f"b {x2} {y2-r+k} {x2-r+k} {y2} {x2-r} {y2} "
            f"l {x1+r} {y2} "
            f"b {x1+r-k} {y2} {x1} {y2-r+k} {x1} {y2-r} "
            f"l {x1} {y1+r} "
            f"b {x1} {y1+r-k} {x1+r-k} {y1} {x1+r} {y1}"
        )

        events.append(f"Dialogue: 0,{s_str},{e_str},BgBox,,0,0,0,,{{\\pos(0,0)\\p1\\1c&H000000&\\3c&H000000&\\bord0\\shad0\\1a&H15&}}{path}{{\\p0}}\n")
        events.append(f"Dialogue: 1,{s_str},{e_str},SubText,,0,0,0,,{{\\pos({cx},{cy})\\an5}}{wrapped_vi}\n")

    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "".join(events))

    print(f"[OK] [Dubber] Đã tạo file phụ đề ASS bo tròn capsule: {out_ass_path.name}")
    return out_audio_path, out_ass_path
