import sys
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yt_dlp

from core.common import FFMPEG, get_media_duration, ffmpeg_location_for_ytdlp
from core.srt_handler import parse_srt, save_srt


# ---------------------------------------------------------------------------
# Cấu hình tải tối ưu cho CDN Bilibili (Akamai)
# ---------------------------------------------------------------------------
# File video Bilibili có thể lên tới 7GB (1080p AVC). CDN Akamai hay cắt kết nối
# HTTP GET dài -> yt-dlp báo:
#     "Got error: 11247223 bytes read, 6534429647 more expected"
# Nguyên nhân: tải 1 luồng không giới hạn, không có Range request chia chunk,
# và số lần retry quá thấp.
#
# Khắc phục:
#   - http_chunk_size : chia nhỏ thành nhiều HTTP Range request -> 1 request chết
#                       không làm hỏng cả file, chỉ retry lại chunk đó.
#   - retries/fragment_retries : tăng lên 20 lần cho file hàng GB.
#   - continuedl : tiếp tục file .part đã tải dở nếu bị ngắt.
HTTP_CHUNK_SIZE = 10 * 1024 * 1024   # 10MB mỗi Range request
DOWNLOAD_RETRIES = 20
FRAGMENT_RETRIES = 20

# User-Agent giả lập Chrome để CDN không chặn
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "Referer": "https://www.bilibili.com/",
    "User-Agent": DEFAULT_UA,
}


def detect_bilibili_browser_cookies() -> Optional[Tuple[str, str]]:
    """
    Tự động tìm cookie đăng nhập Bilibili trong các trình duyệt trên máy.

    Thứ tự ưu tiên:
        1. Firefox  -> đọc sqlite trực tiếp, KHÔNG bị khóa DPAPI (chạy được cả khi
           browser đang mở).
        2. Chrome / Edge -> cần đóng browser để giải mã cookie (DPAPI).

    Trả về tuple (browser, profile) nếu tìm thấy cookie có `SESSDATA`,
    ngược lại trả về None.
    """
    try:
        from yt_dlp.cookies import extract_cookies_from_browser
    except Exception:
        return None

    candidates: List[Tuple[str, str]] = []

    # 1. Firefox (ưu tiên cao nhất)
    ff_root = Path(os.environ.get("APPDATA", "")) / "Mozilla" / "Firefox" / "Profiles"
    if ff_root.exists():
        try:
            for prof_dir in sorted(ff_root.iterdir()):
                if prof_dir.is_dir() and (prof_dir / "cookies.sqlite").exists():
                    candidates.append(("firefox", prof_dir.name))
        except OSError:
            pass

    # 2. Chrome / Edge family
    for br, root in (
        ("chrome", Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"),
        ("edge", Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Edge" / "User Data"),
        ("brave", Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware" / "Brave-Browser" / "User Data"),
    ):
        if not root.exists():
            continue
        try:
            for prof_dir in sorted(root.iterdir()):
                if not prof_dir.is_dir():
                    continue
                has_db = (prof_dir / "Network" / "Cookies").exists() or (prof_dir / "Cookies").exists()
                if has_db:
                    candidates.append((br, prof_dir.name))
        except OSError:
            pass

    # Firefox trước, Chrome-family sau
    candidates.sort(key=lambda x: 0 if x[0] == "firefox" else 1)

    for browser, profile in candidates:
        try:
            jar = extract_cookies_from_browser(browser, profile)
            names = {c.name for c in jar}
            if "SESSDATA" in names:
                print(f"[*] [Bilibili Downloader] Đã tìm thấy cookie Bilibili đăng nhập: {browser}/{profile}")
                return browser, profile
        except Exception:
            # Chrome/Edge hay bị lỗi DPAPI nếu browser đang mở -> bỏ qua, thử profile khác
            continue
    return None


def extract_bvid_and_part(url: str) -> Tuple[str, int]:
    """Trích xuất BV id và số thứ tự tập (?p=X) từ Bilibili URL."""
    bv_match = re.search(r"(BV[a-zA-Z0-9]+)", url)
    bvid = bv_match.group(1) if bv_match else ""
    p_match = re.search(r"[\?\&]p=(\d+)", url)
    part = int(p_match.group(1)) if p_match else 1
    return bvid, part


def clean_bilibili_srt(raw_srt: Path, clean_srt: Path) -> bool:
    """
    Làm sạch file SRT tải từ Bilibili:
    - Loại bỏ các ký tự rác, nốt nhạc ♪♫, ký tự đặc biệt trang trí.
    - Chuẩn hóa ID liên tục từ 1..N.
    - Bỏ qua các câu quá ngắn hoặc trống.
    Trả về True nếu file SRT hợp lệ và có nội dung thoại thực chất (>= 5 câu).
    """
    if not raw_srt.exists() or raw_srt.stat().st_size < 50:
        return False
    try:
        subs = parse_srt(raw_srt)
        valid_subs = []
        for s in subs:
            txt = s["text"].strip(" ♪♫[]()【】\t\r\n")
            if txt and len(txt) >= 1:
                valid_subs.append({
                    "id": len(valid_subs) + 1,
                    "start": s["start"],
                    "end": s["end"],
                    "duration": s["duration"],
                    "text": txt
                })
        if len(valid_subs) < 5:
            return False
        save_srt(valid_subs, clean_srt)
        return True
    except Exception as e:
        print(f"[!] [Bilibili Downloader] Lỗi khi làm sạch SRT: {e}")
        return False


def download_bilibili_bundle(
    url: str,
    out_dir: Path,
    cookies_path: Optional[Path] = None,
    duration_limit: Optional[float] = None,
    part_index: Optional[int] = None,
    browser_for_cookies: Optional[str] = None,
    max_height: int = 1080,
    auto_detect_cookies: bool = True
) -> Dict[str, Any]:
    """
    Tải trọn gói tài nguyên Bilibili chất lượng cao chuẩn Production:
    1. Tự động phát hiện tập (Part/P) trong danh sách phát (Anthology/Collection).
       `part_index` truyền None => lấy `?p=X` từ URL, mặc định 1.
    2. Tự động kiểm tra phụ đề CC mềm tiếng Trung:
       - Tải các ngôn ngữ chuẩn: zh-Hans, zh-CN, zh, ai-zh.
       - Làm sạch qua clean_bilibili_srt (nếu hợp lệ -> dùng luôn, bỏ qua Whisper).
       - Nếu không có phụ đề CC -> zh_srt_path = None (để tự động kích hoạt Whisper Arc B580).
    3. Ưu tiên codec AVC/H.264 hoặc HEVC để tối ưu decode siêu tốc trên Intel Arc QSV.
    4. Trích xuất metadata (Tiê đề, Tags, Mô tả) thành context_hint và lưu context.json.
    5. Hỗ trợ cookies_path / browser_for_cookies để mở khóa 1080p/4K nét căng.
       Tự động dò cookie đăng nhập trong trình duyệt nếu không chỉ định.
    6. Hỗ trợ duration_limit tải nhanh đoạn đầu để test.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bvid, url_p = extract_bvid_and_part(url)
    # Sửa lỗi: trước đây `part_index` mặc định = 1 ghi đè mất `?p=X` lấy từ URL.
    p_num = int(part_index) if part_index else int(url_p or 1)
    clean_url = f"https://www.bilibili.com/video/{bvid}/" if bvid else url

    print(f"[*] [Bilibili Downloader] Đang kết nối Bilibili: {clean_url} (Tập P{p_num})...")

    # 0. Chuẩn bị cookie đăng nhập (mở khóa 1080p / 4K)
    cookie_file: Optional[str] = None
    browser_tuple: Optional[Tuple[str, str]] = None

    if cookies_path and Path(cookies_path).exists():
        cookie_file = str(Path(cookies_path))
        print(f"[*] [Bilibili Downloader] Sử dụng cookies từ file: {cookie_file}")
    elif browser_for_cookies:
        # browser_for_cookies có dạng "chrome" hoặc "chrome:Profile 1"
        if ":" in browser_for_cookies:
            br, prof = browser_for_cookies.split(":", 1)
            browser_tuple = (br.strip(), prof.strip())
        else:
            browser_tuple = (browser_for_cookies.strip(), None)
        print(f"[*] [Bilibili Downloader] Sử dụng cookies từ trình duyệt: {browser_for_cookies}")
    elif auto_detect_cookies:
        browser_tuple = detect_bilibili_browser_cookies()

    # 1. Trích xuất metadata trước (probe)
    probe_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": False,
        "playlist_items": str(p_num),
        "http_headers": dict(DEFAULT_HEADERS),
    }
    if cookie_file:
        probe_opts["cookiefile"] = cookie_file
    elif browser_tuple:
        probe_opts["cookiesfrombrowser"] = browser_tuple

    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info_dict = ydl.extract_info(clean_url, download=False)

    if not info_dict:
        raise RuntimeError(f"Không thể lấy thông tin video từ Bilibili: {url}")

    # Xử lý nếu là anthology / multi-page
    target_info = info_dict
    if "entries" in info_dict and info_dict["entries"]:
        entries = [e for e in info_dict["entries"] if e is not None]
        if entries:
            idx = min(len(entries) - 1, max(0, p_num - 1))
            target_info = entries[idx]

    title = target_info.get("title") or bvid or f"bilibili_p{p_num}"
    # Khử ký tự đặc biệt trong tiêu đề để đặt tên file an toàn trên Windows
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).strip()
    description = target_info.get("description") or ""
    tags = target_info.get("tags") or []
    vid_id = target_info.get("id") or bvid or "video"

    # Biên soạn context_hint chất lượng cao phục vụ Gemini dịch thuật
    hint_parts = [f"Phim/Video: {title}"]
    if tags:
        hint_parts.append(f"Thể loại: {', '.join(tags[:8])}")
    if description:
        clean_desc = description.replace("\r", " ").replace("\n", " ").strip()
        if len(clean_desc) > 150:
            clean_desc = clean_desc[:150] + "..."
        hint_parts.append(f"Mô tả: {clean_desc}")
    context_hint = " | ".join(hint_parts)

    print(f"[*] [Bilibili Downloader] Tiêu đề: {title}")
    print(f"[*] [Bilibili Downloader] Ngữ cảnh: {context_hint}")

    # 2. Cấu hình tải video & phụ đề CC mềm
    format_selector = (
        f"bestvideo[height<={max_height}][vcodec^=avc]+bestaudio/"
        f"bestvideo[height<={max_height}][vcodec^=hev]+bestaudio/"
        f"bestvideo[height<={max_height}]+bestaudio/best[ext=mp4]/best"
    )
    out_video_tmpl = str(out_dir / f"{safe_title}.%(ext)s")

    # `ffmpeg_location` phải trỏ tới nơi yt-dlp tìm được BOTH ffmpeg & ffprobe.
    # Ngoài ra yt-dlp kiểm tra `FFmpegFD.available()` qua PATH (không đọc
    # ffmpeg_location) -> đã được xử lý trong core/common.ensure_ffmpeg_bin().
    ffmpeg_loc = ffmpeg_location_for_ytdlp()

    ydl_opts: Dict[str, Any] = {
        "format": format_selector,
        "merge_output_format": "mp4",
        "outtmpl": out_video_tmpl,
        "noplaylist": False,
        "playlist_items": str(p_num),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["zh-Hans", "zh-CN", "zh", "ai-zh"],
        "subtitlesformat": "srt/vtt/best",
        "writethumbnail": True,
        "writeinfojson": True,
        "concurrent_fragment_downloads": 8,
        "quiet": False,
        "no_warnings": True,
        # --- Khắc phục lỗi CDN cắt kết nối trên file nhiều GB ---
        "http_chunk_size": HTTP_CHUNK_SIZE,
        "continuedl": True,
        "retries": DOWNLOAD_RETRIES,
        "fragment_retries": FRAGMENT_RETRIES,
        "extractor_retries": 5,
        "file_access_retries": 3,
        "socket_timeout": 30,
        "http_headers": dict(DEFAULT_HEADERS),
    }
    if ffmpeg_loc:
        ydl_opts["ffmpeg_location"] = ffmpeg_loc

    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    elif browser_tuple:
        ydl_opts["cookiesfrombrowser"] = browser_tuple

    if duration_limit and duration_limit > 0:
        print(f"[*] [Bilibili Downloader] Giới hạn thời lượng tải: {duration_limit:.1f}s ({duration_limit/60:.1f} phút)...")
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, duration_limit)])

    # Tiến hành tải
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])
    except Exception as e:
        print(f"[!] [Bilibili Downloader] Lỗi tải: {e}")
        # Nếu lỗi do giới hạn thời lượng (cần ffmpeg) -> thử lại không giới hạn
        if duration_limit and duration_limit > 0:
            print("[*] [Bilibili Downloader] Thử lại tải toàn bộ (bỏ giới hạn thời lượng)...")
            ydl_opts.pop("download_ranges", None)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([clean_url])
        else:
            raise

    # 3. Tìm file video tải về
    video_path = None
    for cand in out_dir.glob(f"{safe_title}*.mp4"):
        if not cand.name.endswith(".part"):
            video_path = cand
            break

    if not video_path:
        for cand in out_dir.glob(f"{safe_title}*.mkv"):
            if not cand.name.endswith(".part"):
                video_path = cand
                break

    if not video_path:
        # Fallback: file media mới nhất trong out_dir
        cands = sorted(
            [f for f in out_dir.glob("*.mp4") if not f.name.endswith(".part") and safe_title[:10] in f.name],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        if not cands:
            cands = sorted(
                [f for f in out_dir.glob("*.mp4") if not f.name.endswith(".part")],
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
        if cands:
            video_path = cands[0]

    if not video_path or not video_path.exists():
        raise RuntimeError("Không tìm thấy file video sau khi tải từ Bilibili!")

    print(f"[OK] [Bilibili Downloader] Tải video thành công: {video_path.name} ({video_path.stat().st_size / (1024**2):.1f} MB)")

    # 4. Kiểm tra phụ đề tiếng Trung tải kèm
    zh_srt_path = None
    # Tìm file srt trực tiếp
    for lang in ["zh-Hans", "zh-CN", "zh", "ai-zh"]:
        for cand_sub in out_dir.glob(f"{safe_title}*{lang}*.srt"):
            clean_target = out_dir / f"{safe_title}_zh.srt"
            if clean_bilibili_srt(cand_sub, clean_target):
                zh_srt_path = clean_target
                break
        if zh_srt_path:
            break

    # Nếu chưa thấy, thử bất kỳ file .srt nào của video
    if not zh_srt_path:
        for cand_sub in out_dir.glob(f"{safe_title}*.srt"):
            if cand_sub.name.endswith("_zh.srt"):
                zh_srt_path = cand_sub
                break
            clean_target = out_dir / f"{safe_title}_zh.srt"
            if clean_bilibili_srt(cand_sub, clean_target):
                zh_srt_path = clean_target
                break

    # Nếu có file VTT, chuyển sang SRT bằng ffmpeg và clean
    if not zh_srt_path:
        for cand_vtt in out_dir.glob(f"{safe_title}*.vtt"):
            converted_srt = cand_vtt.with_suffix(".srt")
            subprocess.run([
                FFMPEG, "-nostdin", "-y",
                "-i", str(cand_vtt),
                str(converted_srt)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clean_target = out_dir / f"{safe_title}_zh.srt"
            if converted_srt.exists() and clean_bilibili_srt(converted_srt, clean_target):
                zh_srt_path = clean_target
                break

    if zh_srt_path:
        print(f"🎉 [OK] [Bilibili Downloader] Đã tìm thấy & làm sạch phụ đề CC Bilibili: {zh_srt_path.name} (Bỏ qua Whisper!)")
    else:
        print("[*] [Bilibili Downloader] Không có phụ đề CC mềm Bilibili. Sẽ sử dụng Intel Arc B580 OpenVINO để bóc tách.")

    # 5. Tìm thumbnail
    thumbnail_path = None
    for ext in ("jpg", "png", "webp"):
        for cand_thumb in out_dir.glob(f"{safe_title}*.{ext}"):
            thumbnail_path = cand_thumb
            break
        if thumbnail_path:
            break

    # 6. Lưu file context metadata
    context_data = {
        "title": title,
        "description": description,
        "tags": tags,
        "context_hint": context_hint,
        "url": url,
        "part_index": p_num
    }
    context_file = out_dir / f"{safe_title}_context.json"
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f, ensure_ascii=False, indent=2)

    return {
        "video_path": video_path,
        "zh_srt_path": zh_srt_path,
        "thumbnail_path": thumbnail_path,
        "context_file": context_file,
        "title": title,
        "description": description,
        "tags": tags,
        "context_hint": context_hint,
        "duration": get_media_duration(video_path)
    }
