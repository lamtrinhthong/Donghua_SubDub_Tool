#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastLab Donghua SubDub Studio Tool (Chất lượng Production)
=============================================================
Công cụ đóng gói chuẩn công nghiệp để bóc tách, dịch thuật, lồng tiếng và sản xuất video hoạt hình 3D Donghua:
1. extract   : Tách phụ đề/giọng nói tiếng Trung từ video thành file .srt chuẩn.
2. translate : Dịch file .srt tiếng Trung sang .srt tiếng Việt (tu tiên súc tích, giữ nguyên timing).
3. dub       : Đọc file .srt tiếng Việt -> tổng hợp giọng đọc chuẩn Micro-PID và phụ đề Dynamic Capsule.
4. render    : Làm mờ sub tiếng Hoa ở tầng video, dán sub Việt capsule, gắn icon, muxing audio chất lượng cao.
5. auto      : Tự động hóa toàn bộ quy trình từ Video gốc -> Video thành phẩm (có tùy chọn dừng để người dùng sửa file SRT).
"""

import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from core.common import DEFAULT_ICON, get_media_duration
from core.srt_handler import parse_srt, save_srt
from core.extractor import extract_subtitles_to_srt
from core.translator import translate_srt_file
from core.dubber import dub_from_srt
from core.renderer import render_final_video
from core.bilibili_downloader import download_bilibili_bundle

def main():
    parser = argparse.ArgumentParser(
        description="FastLab Donghua SubDub Studio Tool (Production Standard)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # 1. Tách phụ đề tiếng Trung ra file SRT:
  python cli.py extract --video input.mp4 --out subs_zh.srt

  # 2. Dịch file SRT tiếng Trung sang tiếng Việt:
  python cli.py translate --in subs_zh.srt --out subs_vi.srt

  # 3. Lồng tiếng và sinh phụ đề Dynamic Capsule từ file SRT tiếng Việt:
  python cli.py dub --in subs_vi.srt --audio voice.wav --ass subs.ass --duration 1800

  # 4. Render video hoàn thiện:
  python cli.py render --video input.mp4 --audio voice.wav --ass subs.ass --out final.mp4

  # 5. Chạy toàn bộ quy trình tự động từ A -> Z:
  python cli.py auto --video input.mp4 --out final.mp4 --pause-for-edit
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực hiện")

    # Command: extract
    p_ext = subparsers.add_parser("extract", help="Tách phụ đề tiếng Trung ra file .srt (Tối ưu hóa Intel Arc B580 XMX)")
    p_ext.add_argument("--video", type=str, required=True, help="Đường dẫn video gốc")
    p_ext.add_argument("--out", type=str, required=True, help="Đường dẫn file .srt đầu ra")
    p_ext.add_argument("--duration", type=float, default=None, help="Thời lượng cắt (giây)")
    p_ext.add_argument("--model", type=str, default="large-v3-turbo", help="Model Whisper (large-v3-turbo/small/base)")
    p_ext.add_argument("--backend", type=str, default="auto", choices=["auto", "openvino", "faster-whisper"], help="Engine ASR (auto: ưu tiên Intel Arc B580 OpenVINO)")

    # Command: translate
    p_trans = subparsers.add_parser("translate", help="Dịch file .srt tiếng Trung sang .srt tiếng Việt")
    p_trans.add_argument("--in", dest="in_srt", type=str, required=True, help="File .srt tiếng Trung")
    p_trans.add_argument("--out", dest="out_srt", type=str, required=True, help="File .srt tiếng Việt đầu ra")
    p_trans.add_argument("--model", type=str, default="gemini-3.8-flash-low", help="Model LLM dịch thuật")
    p_trans.add_argument("--context-hint", type=str, default="", help="Ngữ cảnh phim & từ khóa nhân vật từ Bilibili")

    # Command: bilibili
    p_bili = subparsers.add_parser("bilibili", help="Tải video chất lượng cao và phụ đề CC từ Bilibili")
    p_bili.add_argument("--url", type=str, required=True, help="Link video Bilibili (BV...)")
    p_bili.add_argument("--out-dir", type=str, default="workspace/bilibili_download", help="Thư mục lưu video")
    p_bili.add_argument("--cookies", type=str, default="assets/bilibili_cookies.txt", help="File cookies Bilibili (tùy chọn)")
    p_bili.add_argument("--duration", type=float, default=None, help="Giới hạn thời lượng tải (giây)")
    p_bili.add_argument("--part", type=int, default=None,
                         help="Số thứ tự tập (P). Bỏ trống = lấy từ ?p= trong URL, mặc định 1")

    # Command: dub
    p_dub = subparsers.add_parser("dub", help="Lồng tiếng và sinh phụ đề ASS từ file .srt tiếng Việt")
    p_dub.add_argument("--in", dest="in_srt", type=str, required=True, help="File .srt tiếng Việt")
    p_dub.add_argument("--audio", type=str, required=True, help="Đầu ra file master audio (.wav)")
    p_dub.add_argument("--ass", type=str, required=True, help="Đầu ra file phụ đề Dynamic Capsule (.ass)")
    p_dub.add_argument("--duration", type=float, default=0.0, help="Tổng thời lượng video (giây)")
    p_dub.add_argument("--threads", type=int, default=8, help="Số luồng TTS")
    p_dub.add_argument("--voice", type=str, default="nam_minh", choices=["nam_minh", "nam_minh_tram", "hoai_my", "hoai_my_tre", "google_female"], help="Preset giọng đọc (nam_minh/nam_minh_tram/hoai_my/hoai_my_tre/google_female)")

    # Command: render
    p_rnd = subparsers.add_parser("render", help="Render video hoàn chỉnh (blur sub Trung, capsule sub Việt, icon, bitrate cao)")
    p_rnd.add_argument("--video", type=str, required=True, help="Video đầu vào")
    p_rnd.add_argument("--audio", type=str, required=True, help="File audio thuyết minh (.wav)")
    p_rnd.add_argument("--ass", type=str, default=None, help="File phụ đề (.ass, tùy chọn)")
    p_rnd.add_argument("--zh-srt", type=str, default=None, help="File .srt tiếng Trung để tạo Dynamic Blur mask")
    p_rnd.add_argument("--icon", type=str, default=str(DEFAULT_ICON), help="File icon kênh PNG")
    p_rnd.add_argument("--out", type=str, required=True, help="File video hoàn chỉnh đầu ra")
    p_rnd.add_argument("--bitrate", type=int, default=5500, help="Bitrate video (kbps)")
    p_rnd.add_argument("--duration", type=float, default=None, help="Giới hạn thời lượng (giây)")
    p_rnd.add_argument("--no-sub", action="store_true", default=False, help="Tắt dán phụ đề tiếng Việt (mặc định: False, có dán sub Việt)")
    p_rnd.add_argument("--with-sub", dest="no_sub", action="store_false", help="Bật dán phụ đề tiếng Việt (mặc định)")
    p_rnd.add_argument("--blur-zh", action="store_true", default=False, help="Bật làm mờ sub tiếng Trung (Mặc định: False)")

    # Command: auto (Toàn bộ quy trình)
    p_all = subparsers.add_parser("auto", help="Chạy tự động từ A -> Z (Video -> SRT -> Audio -> Video hoàn thiện)")
    p_all.add_argument("--video", type=str, required=True, help="Video đầu vào (hoặc URL Bilibili)")
    p_all.add_argument("--out", type=str, required=True, help="Video hoàn chỉnh đầu ra")
    p_all.add_argument("--duration", type=float, default=None, help="Giới hạn thời lượng (giây)")
    p_all.add_argument("--voice", type=str, default="nam_minh", choices=["nam_minh", "nam_minh_tram", "hoai_my", "hoai_my_tre", "google_female"], help="Preset giọng đọc (nam_minh/nam_minh_tram/hoai_my/hoai_my_tre/google_female)")
    p_all.add_argument("--bitrate", type=int, default=5500, help="Bitrate video đầu ra kbps (Mặc định: 5500)")
    p_all.add_argument("--pause-for-edit", action="store_true", help="Tạm dừng sau khi dịch để người dùng mở file SRT chỉnh sửa")
    p_all.add_argument("--no-sub", action="store_true", default=False, help="Tắt dán phụ đề tiếng Việt (mặc định: False, có dán sub Việt)")
    p_all.add_argument("--with-sub", dest="no_sub", action="store_false", help="Bật dán phụ đề tiếng Việt (mặc định)")
    p_all.add_argument("--blur-zh", action="store_true", default=False, help="Bật làm mờ sub tiếng Trung (Mặc định: False)")

    # Command: menu (Menu tương tác trực quan)
    subparsers.add_parser("menu", help="Mở giao diện menu tương tác trực quan để chọn và dán input")

    args = parser.parse_args()

    if not args.command or args.command == "menu":
        from menu import main_menu
        main_menu()
        return

    if args.command == "extract":
        extract_subtitles_to_srt(
            Path(args.video),
            Path(args.out),
            model_size=args.model,
            backend=args.backend,
            duration=args.duration
        )

    elif args.command == "translate":
        translate_srt_file(
            Path(args.in_srt),
            Path(args.out_srt),
            model=args.model,
            context_hint=args.context_hint
        )

    elif args.command == "bilibili":
        download_bilibili_bundle(
            url=args.url,
            out_dir=Path(args.out_dir),
            cookies_path=Path(args.cookies) if args.cookies else None,
            duration_limit=args.duration,
            part_index=args.part
        )

    elif args.command == "dub":
        dur = args.duration
        if dur <= 0:
            subs = parse_srt(Path(args.in_srt))
            dur = subs[-1]["end"] + 2.0 if subs else 60.0
        dub_from_srt(
            Path(args.in_srt),
            Path(args.audio),
            Path(args.ass),
            total_duration=dur,
            threads=args.threads,
            voice_id=args.voice
        )

    elif args.command == "render":
        render_final_video(
            source_video=Path(args.video),
            voice_track=Path(args.audio),
            ass_path=Path(args.ass) if args.ass else None,
            zh_srt_path=Path(args.zh_srt) if args.zh_srt else None,
            channel_icon=Path(args.icon) if args.icon else None,
            out_video=Path(args.out),
            duration=args.duration,
            bitrate_kbps=args.bitrate,
            no_sub=args.no_sub,
            blur_zh=args.blur_zh
        )

    elif args.command == "auto":
        v_out = Path(args.out)
        video_arg = args.video
        context_hint = ""
        srt_zh = None

        # Tự động tải từ Bilibili nếu là đường dẫn web
        if video_arg.startswith(("http://", "https://")) or "bilibili.com" in video_arg or video_arg.startswith("BV"):
            bili_dir = v_out.parent / "bilibili_raw"
            bundle = download_bilibili_bundle(
                url=video_arg,
                out_dir=bili_dir,
                cookies_path=Path("assets/bilibili_cookies.txt"),
                duration_limit=args.duration
            )
            v_in = bundle["video_path"]
            srt_zh = bundle.get("zh_srt_path")
            context_hint = bundle.get("context_hint", "")
        else:
            v_in = Path(video_arg)

        work_dir = v_out.parent / f"work_{v_in.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        srt_vi = work_dir / f"{v_in.stem}_vi.srt"
        voice_wav = work_dir / f"{v_in.stem}_voice.wav"
        ass_file = work_dir / f"{v_in.stem}_subtitles.ass"

        dur = args.duration or get_media_duration(v_in)

        print("\n" + "=" * 70)
        print("KHỞI CHẠY AUTO PIPELINE (PRODUCTION STANDARD)")
        print(f"Video nguồn: {v_in.name} ({dur:.1f}s)")
        print(f"Thư mục làm việc: {work_dir}")
        print("=" * 70)

        # Bước 1: Extract (Tự động quyết định có chạy Whisper hay không)
        if not srt_zh:
            srt_zh = work_dir / f"{v_in.stem}_zh.srt"
            if srt_zh.exists() and srt_zh.stat().st_size > 100:
                print(f"\n>>> BƯỚC 1/4: Đã có sẵn phụ đề tiếng Trung ({srt_zh.name}), bỏ qua Whisper!")
            else:
                print("\n>>> BƯỚC 1/4: Bilibili không có sub CC, bật Intel Arc B580 OpenVINO để bóc sub...")
                extract_subtitles_to_srt(v_in, srt_zh, duration=dur, work_dir=work_dir)
        else:
            print(f"\n>>> BƯỚC 1/4: Đã có sẵn phụ đề CC mềm Bilibili ({srt_zh.name}), bỏ qua Whisper!")

        # Bước 2: Translate (Truyền context_hint từ Bilibili)
        if srt_vi.exists() and srt_vi.stat().st_size > 100:
            print(f"\n>>> BƯỚC 2/4: Đã có sẵn file SRT tiếng Việt ({srt_vi.name}), bỏ qua dịch thuật!")
        else:
            print("\n>>> BƯỚC 2/4: Dịch file SRT tiếng Trung sang tiếng Việt...")
            translate_srt_file(srt_zh, srt_vi, context_hint=context_hint)

        if args.pause_for_edit:
            print("\n" + "#" * 70)
            print(f"[THÔNG BÁO] File SRT tiếng Việt đã tạo tại: {srt_vi}")
            print("Bạn có thể mở file này bằng Subtitle Edit, Aegisub hoặc Notepad để sửa.")
            input("Sau khi sửa xong, nhấn [ENTER] để tiếp tục lồng tiếng và render...")
            print("#" * 70 + "\n")

        # Bước 3: Dub
        if voice_wav.exists() and voice_wav.stat().st_size > 500000 and ass_file.exists() and ass_file.stat().st_size > 100:
            print(f"\n>>> BƯỚC 3/4: Đã có sẵn file voice ({voice_wav.name}) và sub ASS, bỏ qua lồng tiếng!")
        else:
            print(f"\n>>> BƯỚC 3/4: Lồng tiếng thuyết minh (Giọng: {args.voice}) và tạo phụ đề Dynamic Capsule...")
            dub_from_srt(srt_vi, voice_wav, ass_file, total_duration=dur, work_dir=work_dir, voice_id=args.voice)

        # Bước 4: Render
        sub_status = "ĐÃ TẮT SUB VIỆT" if args.no_sub else "Dynamic Capsule sub Việt"
        blur_status = "Làm mờ sub Trung" if args.blur_zh else "GIỮ NGUYÊN SUB TRUNG (Không làm mờ)"
        print(f"\n>>> BƯỚC 4/4: Render video chuẩn Studio ({blur_status}, {sub_status})...")
        render_final_video(
            source_video=v_in,
            voice_track=voice_wav,
            ass_path=ass_file,
            zh_srt_path=srt_zh,
            channel_icon=DEFAULT_ICON,
            out_video=v_out,
            duration=dur,
            bitrate_kbps=args.bitrate,
            no_sub=args.no_sub,
            blur_zh=args.blur_zh
        )

        print("\n" + "=" * 70)
        print(f"🎉 TẤT CẢ CÔNG ĐOẠN ĐÃ HOÀN TẤT XUẤT SẮC: {v_out.name}")
        print(f"File video thành phẩm: {v_out}")
        print(f"File phụ đề tiếng Việt SRT: {srt_vi}")
        print("=" * 70)

if __name__ == "__main__":
    main()
