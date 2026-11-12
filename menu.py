import os
import sys
import time
from pathlib import Path
from typing import Optional

# Enable VT100 ANSI sequences on Windows
if os.name == "nt":
    os.system("")

from core.common import DEFAULT_ICON, get_media_duration
from core.bilibili_downloader import download_bilibili_bundle
from core.extractor import extract_subtitles_to_srt
from core.translator import translate_srt_file
from core.dubber import dub_from_srt, VOICE_PRESETS
from core.renderer import render_final_video
from core.srt_handler import parse_srt


# Colors for ANSI terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════════╗
║             FASTLAB AI STUDIO - DONGHUA SUB & DUB AUTOMATION TOOL             ║
║            Hệ Thống Thuyết Minh & Phụ Đề Hoạt Hình 3D Chuẩn Studio            ║
╚═══════════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def clean_str(val: str) -> str:
    """Loại bỏ dấu nháy kép, nháy đơn và khoảng trắng thừa khi kéo thả file từ Windows Explorer."""
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1].strip()
    return val


def prompt_input(label: str, default: Optional[str] = None, must_exist: bool = False) -> str:
    """Nhập chuỗi ký tự với giá trị mặc định và kiểm tra file tồn tại nếu cần."""
    while True:
        if default:
            prompt = f"{BOLD}{label}{RESET} [{YELLOW}{default}{RESET}]: "
        else:
            prompt = f"{BOLD}{label}{RESET}: "

        val = clean_str(input(prompt))
        if not val and default is not None:
            val = default

        if not val:
            print(f"{RED}[!] Vui lòng không để trống trường này.{RESET}")
            continue

        if must_exist:
            p = Path(val)
            if not p.exists():
                print(f"{RED}[!] Không tìm thấy file: '{val}'. Vui lòng kiểm tra lại đường dẫn.{RESET}")
                continue

        return val


def prompt_yes_no(label: str, default: bool = True) -> bool:
    """Hỏi câu hỏi Yes/No với giá trị mặc định."""
    d_str = "Y/n" if default else "y/N"
    while True:
        val = clean_str(input(f"{BOLD}{label}{RESET} [{YELLOW}{d_str}{RESET}]: ")).lower()
        if not val:
            return default
        if val in ("y", "yes", "có", "c", "1"):
            return True
        if val in ("n", "no", "không", "k", "0"):
            return False
        print(f"{RED}[!] Vui lòng nhập Y hoặc N.{RESET}")


def prompt_voice(default_voice: str = "nam_minh") -> str:
    """Hiển thị menu trực quan để người dùng chọn giọng đọc."""
    voice_list = [
        ("nam_minh", "Nam Minh (Microsoft Neural - Trầm ấm, chuẩn kiếm hiệp/tiên hiệp) [Khuyên dùng]"),
        ("nam_minh_tram", "Nam Minh Trầm Sâu (Giọng dày uy lực, phim mạt thế/hành động)"),
        ("hoai_my", "Hoài My (Microsoft Neural - Nữ truyền cảm, tự nhiên)"),
        ("hoai_my_tre", "Hoài My Trẻ Trung (Giọng trong sáng, vui tươi/hoạt hình)"),
        ("google_female", "Chị Google (Nhanh, ổn định)")
    ]

    print(f"\n{CYAN}{BOLD}--- CHỌN GIỌNG ĐỌC THUYẾT MINH ---{RESET}")
    default_idx = 1
    for i, (key, desc) in enumerate(voice_list, 1):
        is_def = " (Mặc định)" if key == default_voice else ""
        if key == default_voice:
            default_idx = i
        print(f"  {BOLD}[{i}]{RESET} {desc}{is_def}")

    while True:
        choice = clean_str(input(f"Chọn giọng [1-{len(voice_list)}, Enter = {default_idx}]: "))
        if not choice:
            return default_voice
        try:
            idx = int(choice)
            if 1 <= idx <= len(voice_list):
                selected = voice_list[idx - 1][0]
                print(f"{GREEN}[OK] Đã chọn giọng: {selected}{RESET}")
                return selected
        except ValueError:
            pass
        print(f"{RED}[!] Lựa chọn không hợp lệ. Vui lòng nhập từ 1 đến {len(voice_list)}.{RESET}")


# ==============================================================================
# CÁC WORKFLOW THỰC THI CHÍNH
# ==============================================================================

def action_auto_pipeline():
    """Workflow 1: Chạy Tự Động Toàn Diện A -> Z."""
    print(f"\n{GREEN}{BOLD}=== [1] CHẠY TỰ ĐỘNG TOÀN DIỆN A -> Z (BILIBILI HOẶC VIDEO MP4) ==={RESET}")
    print(f"{DIM}Hỗ trợ dán trực tiếp link Bilibili (BV...) hoặc kéo thả file video .mp4 từ máy tính.{RESET}\n")

    input_src = prompt_input("Dán link Bilibili hoặc đường dẫn video MP4")
    is_url = input_src.startswith(("http://", "https://")) or "bilibili.com" in input_src or input_src.startswith("BV")

    # Xác định thư mục làm việc và tên file xuất
    if is_url:
        default_out = "workspace/FINAL_DUBBED_BILIBILI.mp4"
    else:
        src_path = Path(input_src)
        default_out = str(src_path.parent / f"FINAL_DUBBED_{src_path.stem}.mp4")

    out_video_str = prompt_input("Đường dẫn file video đầu ra hoàn chỉnh (.mp4)", default=default_out)
    out_video = Path(out_video_str)

    # Chọn giọng đọc
    voice = prompt_voice(default_voice="nam_minh")

    # Tùy chọn thời lượng
    dur_str = clean_str(input(f"{BOLD}Giới hạn thời lượng (giây) [Enter = toàn bộ video, hoặc nhập số vd: 600 cho 10m]{RESET}: "))
    dur_limit = float(dur_str) if dur_str else None

    # Tùy chọn phụ đề & hình ảnh
    with_sub = prompt_yes_no("Bật phụ đề tiếng Việt Capsule bo góc?", default=True)
    blur_zh = prompt_yes_no("Bật làm mờ phụ đề tiếng Trung gốc?", default=False)
    pause_edit = prompt_yes_no("Tạm dừng sau khi dịch để bạn sửa tay file SRT nếu muốn?", default=False)

    print(f"\n{CYAN}{BOLD}--- BẮT ĐẦU KHỞI CHẠY AUTO PIPELINE ---{RESET}")
    t_start = time.time()

    # 1. Chuẩn bị video và SRT gốc
    context_hint = ""
    srt_zh = None
    v_out = out_video
    v_out.parent.mkdir(parents=True, exist_ok=True)

    if is_url:
        print(f"\n{YELLOW}>>> [BƯỚC 1/4] Tải video chất lượng cao từ Bilibili...{RESET}")
        bili_dir = v_out.parent / "bilibili_raw"
        bundle = download_bilibili_bundle(
            url=input_src,
            out_dir=bili_dir,
            cookies_path=Path("assets/bilibili_cookies.txt"),
            duration_limit=dur_limit
        )
        v_in = bundle["video_path"]
        srt_zh = bundle.get("zh_srt_path")
        context_hint = bundle.get("context_hint", "")
    else:
        v_in = Path(input_src)

    work_dir = v_out.parent / f"work_{v_in.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    srt_vi = work_dir / f"{v_in.stem}_vi.srt"
    voice_wav = work_dir / f"{v_in.stem}_voice.wav"
    ass_file = work_dir / f"{v_in.stem}_subtitles.ass"
    dur = dur_limit or get_media_duration(v_in)

    # Bước 1: Trích xuất sub tiếng Trung
    if not srt_zh:
        print(f"\n{YELLOW}>>> [BƯỚC 1/4] Bóc phụ đề tiếng Trung bằng Intel Arc B580 OpenVINO...{RESET}")
        srt_zh = work_dir / f"{v_in.stem}_zh.srt"
        extract_subtitles_to_srt(v_in, srt_zh, duration=dur, work_dir=work_dir)
    else:
        print(f"\n{GREEN}>>> [BƯỚC 1/4] Đã có sẵn phụ đề CC mềm từ Bilibili ({srt_zh.name}), bỏ qua Whisper!{RESET}")

    # Bước 2: Dịch SRT Trung -> Việt
    print(f"\n{YELLOW}>>> [BƯỚC 2/4] Dịch phụ đề sang tiếng Việt qua LLM (Context Hint Bilibili)...{RESET}")
    translate_srt_file(srt_zh, srt_vi, context_hint=context_hint)

    if pause_edit:
        print("\n" + "#" * 70)
        print(f"{YELLOW}[THÔNG BÁO] File SRT tiếng Việt đã tạo tại: {srt_vi}{RESET}")
        print("Bạn có thể mở file này bằng Subtitle Edit, Aegisub hoặc Notepad để sửa.")
        input(f"{BOLD}Sau khi sửa xong, nhấn [ENTER] để tiếp tục lồng tiếng và render...{RESET}")
        print("#" * 70 + "\n")

    # Bước 3: Lồng tiếng thuyết minh
    print(f"\n{YELLOW}>>> [BƯỚC 3/4] Lồng tiếng thuyết minh (Giọng: {voice}) & Micro-PID Warper...{RESET}")
    dub_from_srt(srt_vi, voice_wav, ass_file, total_duration=dur, work_dir=work_dir, voice_id=voice)

    # Bước 4: Render hoàn chỉnh
    sub_status = "Dynamic Capsule sub Việt" if with_sub else "ĐÃ TẮT SUB VIỆT"
    blur_status = "Làm mờ sub Trung" if blur_zh else "GIỮ NGUYÊN SUB TRUNG"
    print(f"\n{YELLOW}>>> [BƯỚC 4/4] Render hoàn thiện ({blur_status}, {sub_status}, Xóa Logo Bilibili, Auto-Ducking)...{RESET}")
    render_final_video(
        source_video=v_in,
        voice_track=voice_wav,
        ass_path=ass_file,
        zh_srt_path=srt_zh,
        channel_icon=DEFAULT_ICON,
        out_video=v_out,
        duration=dur,
        bitrate_kbps=5500,
        no_sub=(not with_sub),
        blur_zh=blur_zh
    )

    t_total = time.time() - t_start
    print(f"\n{GREEN}{BOLD}═══════════════════════════════════════════════════════════════════════════════")
    print(f"🎉 AUTO PIPELINE HOÀN TẤT XUẤT SẮC TRONG {t_total:.1f} GIÂY!")
    print(f"📁 Video thành phẩm: {v_out.resolve()}")
    print(f"═══════════════════════════════════════════════════════════════════════════════{RESET}\n")


def action_download_bilibili():
    """Workflow 2: Tải video từ Bilibili."""
    print(f"\n{GREEN}{BOLD}=== [2] TẢI VIDEO CHẤT LƯỢNG CAO & PHỤ ĐỀ CC TỪ BILIBILI ==={RESET}\n")
    url = prompt_input("Nhập link video Bilibili (BV...)")
    out_dir = prompt_input("Thư mục lưu video", default="workspace/bilibili_download")
    part_str = clean_str(input(f"{BOLD}Số thứ tự tập (P) nếu là playlist [Enter = lấy từ URL ?p= hoặc 1]{RESET}: "))
    part = int(part_str) if part_str else None
    dur_str = clean_str(input(f"{BOLD}Giới hạn thời lượng tải (giây) [Enter = toàn bộ video]{RESET}: "))
    dur = float(dur_str) if dur_str else None

    bundle = download_bilibili_bundle(
        url=url,
        out_dir=Path(out_dir),
        cookies_path=Path("assets/bilibili_cookies.txt"),
        duration_limit=dur,
        part_index=part
    )
    print(f"\n{GREEN}[OK] Đã tải xong video: {bundle['video_path']}{RESET}")
    if bundle.get("zh_srt_path"):
        print(f"{GREEN}[OK] Đã tải kèm phụ đề CC: {bundle['zh_srt_path']}{RESET}")


def action_extract_subtitles():
    """Workflow 3: Bóc phụ đề tiếng Trung bằng Whisper Arc B580."""
    print(f"\n{GREEN}{BOLD}=== [3] BÓC PHỤ ĐỀ TIẾNG TRUNG BẰNG WHISPER (INTEL ARC B580 OPENVINO) ==={RESET}\n")
    v_in = prompt_input("Kéo thả file video nguồn (.mp4/.mkv)", must_exist=True)
    v_path = Path(v_in)
    default_srt = str(v_path.parent / f"{v_path.stem}_zh.srt")
    out_srt = prompt_input("Đường dẫn file .srt đầu ra", default=default_srt)
    dur_str = clean_str(input(f"{BOLD}Giới hạn thời lượng (giây) [Enter = toàn bộ]{RESET}: "))
    dur = float(dur_str) if dur_str else None

    extract_subtitles_to_srt(
        video_path=Path(v_in),
        out_srt_path=Path(out_srt),
        model_size="large-v3-turbo",
        backend="auto",
        duration=dur
    )
    print(f"\n{GREEN}[OK] Bóc sub thành công: {out_srt}{RESET}")


def action_translate_srt():
    """Workflow 4: Dịch phụ đề SRT Trung -> Việt."""
    print(f"\n{GREEN}{BOLD}=== [4] DỊCH PHỤ ĐỀ SRT TRUNG -> VIỆT (LLM GEMINI FLASH) ==={RESET}\n")
    in_srt = prompt_input("Kéo thả file .srt tiếng Trung", must_exist=True)
    p_in = Path(in_srt)
    default_out = str(p_in.parent / f"{p_in.stem.replace('_zh', '')}_vi.srt")
    out_srt = prompt_input("Đường dẫn file .srt tiếng Việt đầu ra", default=default_out)
    context_hint = clean_str(input(f"{BOLD}Ngữ cảnh phim / tên nhân vật / môn phái [Tùy chọn, Enter = bỏ qua]{RESET}: "))

    translate_srt_file(
        input_srt=Path(in_srt),
        output_srt=Path(out_srt),
        model="gemini-3.8-flash-low",
        context_hint=context_hint
    )
    print(f"\n{GREEN}[OK] Dịch SRT thành công: {out_srt}{RESET}")


def action_dub():
    """Workflow 5: Lồng tiếng thuyết minh."""
    print(f"\n{GREEN}{BOLD}=== [5] LỒNG TIẾNG THUYẾT MINH (EDGE NEURAL TTS & MICRO-PID) ==={RESET}\n")
    in_srt = prompt_input("Kéo thả file .srt tiếng Việt", must_exist=True)
    p_srt = Path(in_srt)
    default_wav = str(p_srt.parent / f"{p_srt.stem}_voice.wav")
    default_ass = str(p_srt.parent / f"{p_srt.stem}_subtitles.ass")

    out_audio = prompt_input("File audio thuyết minh (.wav) đầu ra", default=default_wav)
    out_ass = prompt_input("File phụ đề ASS Dynamic Capsule đầu ra", default=default_ass)
    voice = prompt_voice(default_voice="nam_minh")

    subs = parse_srt(Path(in_srt))
    calc_dur = subs[-1]["end"] + 2.0 if subs else 60.0
    dur_str = clean_str(input(f"{BOLD}Tổng thời lượng video (giây) [Enter = tự tính {calc_dur:.1f}s]{RESET}: "))
    dur = float(dur_str) if dur_str else calc_dur

    dub_from_srt(
        input_srt_path=Path(in_srt),
        out_audio_path=Path(out_audio),
        out_ass_path=Path(out_ass),
        total_duration=dur,
        voice_id=voice,
        threads=8
    )
    print(f"\n{GREEN}[OK] Lồng tiếng thành công! File audio: {out_audio}{RESET}")


def action_render():
    """Workflow 6: Render video hoàn chỉnh."""
    print(f"\n{GREEN}{BOLD}=== [6] RENDER VIDEO HOÀN CHỈNH CHUẨN STUDIO ==={RESET}\n")
    v_in = prompt_input("Kéo thả file video gốc", must_exist=True)
    a_in = prompt_input("Kéo thả file audio thuyết minh (.wav)", must_exist=True)
    ass_in = clean_str(input(f"{BOLD}Kéo thả file phụ đề ASS [Enter = không dán sub]{RESET}: "))
    ass_path = Path(ass_in) if (ass_in and Path(ass_in).exists()) else None

    p_vid = Path(v_in)
    default_out = str(p_vid.parent / f"FINAL_{p_vid.stem}.mp4")
    out_vid = prompt_input("File video hoàn thiện đầu ra (.mp4)", default=default_out)

    with_sub = (ass_path is not None) and prompt_yes_no("Bật dán phụ đề tiếng Việt?", default=True)
    blur_zh = prompt_yes_no("Bật làm mờ phụ đề tiếng Trung gốc?", default=False)
    zh_srt = None
    if blur_zh:
        zh_in = clean_str(input(f"{BOLD}Kéo thả file SRT tiếng Trung để định vị blur [Tùy chọn]{RESET}: "))
        if zh_in and Path(zh_in).exists():
            zh_srt = Path(zh_in)

    dur = get_media_duration(Path(v_in))
    render_final_video(
        source_video=Path(v_in),
        voice_track=Path(a_in),
        ass_path=ass_path,
        zh_srt_path=zh_srt,
        channel_icon=DEFAULT_ICON,
        out_video=Path(out_vid),
        duration=dur,
        bitrate_kbps=5500,
        no_sub=(not with_sub),
        blur_zh=blur_zh
    )
    print(f"\n{GREEN}[OK] Render hoàn tất: {out_vid}{RESET}")


# ==============================================================================
# MENU CHÍNH
# ==============================================================================

def main_menu():
    """Vòng lặp menu tương tác chính của Tool."""
    while True:
        print_banner()
        print(f" {BOLD}{CYAN}[1]{RESET} {BOLD}⚡ Chạy TỰ ĐỘNG TOÀN DIỆN A -> Z{RESET} (Link Bilibili hoặc Video MP4)")
        print(f" {BOLD}{CYAN}[2]{RESET} 📥 Tải Video chất lượng cao & Phụ đề CC từ Bilibili")
        print(f" {BOLD}{CYAN}[3]{RESET} 🎙️ Bóc phụ đề tiếng Trung bằng Whisper (Intel Arc B580 OpenVINO)")
        print(f" {BOLD}{CYAN}[4]{RESET} 🌐 Dịch phụ đề SRT Trung -> Việt (LLM Gemini Flash)")
        print(f" {BOLD}{CYAN}[5]{RESET} 🗣️ Lồng tiếng thuyết minh (Edge Neural TTS: Nam Minh, Hoài My...)")
        print(f" {BOLD}{CYAN}[6]{RESET} 🎬 Render Video Studio (Xóa Watermark Bilibili, Capsule Sub, Ducking)")
        print(f" {BOLD}{RED}[0]{RESET} ❌ Thoát")
        print("-" * 79)

        choice = clean_str(input(f"{BOLD}Vui lòng chọn tác vụ [0-6]: {RESET}"))

        if choice == "1":
            try:
                action_auto_pipeline()
            except KeyboardInterrupt:
                print(f"\n{YELLOW}[!] Đã tạm dừng tác vụ.{RESET}")
            except Exception as e:
                print(f"\n{RED}[!] Lỗi trong quá trình xử lý: {e}{RESET}")
        elif choice == "2":
            try:
                action_download_bilibili()
            except Exception as e:
                print(f"\n{RED}[!] Lỗi: {e}{RESET}")
        elif choice == "3":
            try:
                action_extract_subtitles()
            except Exception as e:
                print(f"\n{RED}[!] Lỗi: {e}{RESET}")
        elif choice == "4":
            try:
                action_translate_srt()
            except Exception as e:
                print(f"\n{RED}[!] Lỗi: {e}{RESET}")
        elif choice == "5":
            try:
                action_dub()
            except Exception as e:
                print(f"\n{RED}[!] Lỗi: {e}{RESET}")
        elif choice == "6":
            try:
                action_render()
            except Exception as e:
                print(f"\n{RED}[!] Lỗi: {e}{RESET}")
        elif choice in ("0", "q", "exit", "quit"):
            print(f"\n{CYAN}Cảm ơn bạn đã sử dụng FastLab AI Studio Tool! Hẹn gặp lại! 👋{RESET}\n")
            break
        else:
            print(f"\n{RED}[!] Lựa chọn không hợp lệ, vui lòng chọn số từ 0 đến 6.{RESET}")

        cont = clean_str(input(f"\n{DIM}Nhấn [ENTER] để quay lại Menu chính (hoặc gõ '0' để thoát): {RESET}"))
        if cont in ("0", "q", "exit"):
            print(f"\n{CYAN}Tạm biệt! 👋{RESET}\n")
            break
        print("\n" * 2)


if __name__ == "__main__":
    main_menu()
