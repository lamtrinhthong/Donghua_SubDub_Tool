import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional
import imageio_ffmpeg

# Single Source of Truth cho FFMPEG binary
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Thư mục chứa tools dùng chung cho project (được git-ignore)
TOOLS_DIR = Path(__file__).resolve().parent.parent / ".tools"
FFMPEG_BIN_DIR = TOOLS_DIR / "bin"

# Đường dẫn icon kênh chính thức mặc định (nằm ngay trong thư mục assets của project)
DEFAULT_ICON = Path(__file__).resolve().parent.parent / "assets" / "channel_icon.png"


def ensure_ffmpeg_bin() -> Path:
    """
    Chuẩn bị một bản `ffmpeg.exe` nằm trong thư mục có tên chuẩn và thêm thư mục đó vào PATH.

    Lý do cần thiết:
        yt-dlp kiểm tra khả năng tải 1 phần video qua `FFmpegFD.available()`, hàm này
        KHÔNG đọc tham số `ffmpeg_location` mà chỉ dò trong `PATH`
        (xem yt_dlp/downloader/external.py -> `FFmpegFD.available()`).
        Vì vậy nếu chỉ truyền `ffmpeg_location` (đường dẫn file do imageio_ffmpeg cung cấp
        với tên lạ như `ffmpeg-win-x86_64-v9.0.2.exe`) thì yt-dlp sẽ báo:
            "You have requested downloading the video partially, but ffmpeg is not installed"

    Hàm này copy binary ffmpeg hiện có sang `<project>/.tools/bin/ffmpeg.exe` (tên chuẩn)
    rồi chèn thư mục đó vào `os.environ["PATH"]` để cả yt-dlp lẫn các tool khác tìm thấy.

    Trả về: Path tới file ffmpeg.exe đã chuẩn bị.
    """
    FFMPEG_BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = FFMPEG_BIN_DIR / "ffmpeg.exe"

    try:
        src = Path(FFMPEG)
        need_copy = True
        if target.exists():
            try:
                need_copy = target.stat().st_size != src.stat().st_size
            except OSError:
                need_copy = True
        if need_copy:
            shutil.copy2(src, target)
    except Exception as e:
        print(f"[!] [Common] Không thể chuẩn bị ffmpeg.exe: {e}")
        return Path(FFMPEG)

    # Chèn thư mục chứa ffmpeg.exe vào PATH (ưu tiên cao nhất)
    bin_dir = str(FFMPEG_BIN_DIR)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if bin_dir.lower() not in [p.lower() for p in parts]:
        os.environ["PATH"] = bin_dir + os.pathsep + current

    return target


def ffmpeg_location_for_ytdlp() -> Optional[str]:
    """
    Trả về giá trị nên truyền cho tham số `ffmpeg_location` của yt-dlp.

    - Nếu tìm thấy `ffprobe.exe` cùng thư mục với ffmpeg -> truyền THƯ MỤC
      (yt-dlp sẽ tự ghép `ffmpeg` + `ffprobe` trong thư mục đó).
    - Ngược lại -> truyền đường dẫn file `ffmpeg.exe` (yt-dlp tự suy ra `ffprobe` từ tên file).
    - Không tìm thấy -> None (để yt-dlp tự dò trong PATH).
    """
    ensure_ffmpeg_bin()
    if (FFMPEG_BIN_DIR / "ffprobe.exe").exists():
        return str(FFMPEG_BIN_DIR)
    ffmpeg_exe = FFMPEG_BIN_DIR / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        return str(ffmpeg_exe)
    return None

def seconds_to_srt_time(seconds: float) -> str:
    """
    Chuyển đổi giây thành định dạng SRT: HH:MM:SS,mmm
    Sử dụng divmod chuẩn để chống hoàn toàn lỗi tràn giây 60 (vd: 59.999 + 1ms -> 60.000).
    """
    total_millis = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def srt_time_to_seconds(time_str: str) -> float:
    """
    Chuyển đổi chuỗi thời gian SRT (HH:MM:SS,mmm hoặc HH:MM:SS.mmm) thành giây chuẩn (float).
    Tự động chuẩn hóa mili-giây đủ 3 chữ số kể cả định dạng rút gọn.
    """
    time_str = time_str.strip().replace(".", ",")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split(",")
    seconds = int(sec_parts[0])
    millis = int(sec_parts[1].ljust(3, "0")[:3]) if len(sec_parts) > 1 else 0
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

def format_ass_time(seconds: float) -> str:
    """
    Chuyển đổi giây thành định dạng thời gian ASS: H:MM:SS.cc (centiseconds - 2 chữ số thập phân).
    """
    total_centis = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total_centis, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, centis = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"

def get_media_duration(file_path: Path) -> float:
    """
    Đo thời lượng của bất kỳ file media (mp4, mkv, wav, mp3) qua FFmpeg probe.
    """
    if not file_path.exists():
        return 0.0
    cmd = [FFMPEG, "-nostdin", "-i", str(file_path)]
    p = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="ignore")
    for line in p.stderr.splitlines():
        if "Duration:" in line:
            try:
                parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except Exception:
                pass
    return 0.0

def get_wav_duration(wav_path: Path) -> float:
    """
    Đo thời lượng file WAV cực nhanh qua header (không cần gọi subprocess ffmpeg).
    """
    if not wav_path.exists():
        return 0.0
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 3)
    except Exception:
        return get_media_duration(wav_path)
