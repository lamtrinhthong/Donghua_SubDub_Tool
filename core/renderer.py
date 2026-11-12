import sys
import os
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional
from core.common import FFMPEG, get_media_duration
from core.dynamic_blur import generate_dynamic_blur_ass, detect_static_corner_watermarks


def render_final_video(
    source_video: Path,
    voice_track: Path,
    ass_path: Optional[Path] = None,
    zh_srt_path: Optional[Path] = None,  # Truyền file SRT gốc (hoặc SRT tiếng Việt) để lấy mốc câu và độ dài
    channel_icon: Optional[Path] = None,
    out_video: Path = None,
    duration: Optional[float] = None,
    bitrate_kbps: int = 5500,
    preset: str = "fast",
    no_sub: bool = False,
    blur_zh: bool = False
) -> Path:
    """
    Render video chuẩn Production:
    1. Làm mờ sub tiếng Trung (Tùy chọn, mặc định: False): Chỉ kích hoạt khi blur_zh=True.
    2. Gắn Icon kênh góc trái (20:20).
    3. (Tùy chọn) In phụ đề tiếng Việt Dynamic Rounded Capsule (nếu no_sub=False).
    4. Auto-Ducking âm thanh nền + giọng đọc tiếng Việt 48kHz.
    """
    out_video.parent.mkdir(parents=True, exist_ok=True)
    temp_render = (out_video.parent / f"temp_{out_video.name}").resolve()
    src_abs = Path(source_video).resolve()
    voice_abs = Path(voice_track).resolve()

    filter_parts = [
        "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black[vscaled]"
    ]

    # 1. Tự động dò tìm và xóa sạch Logo Bilibili + Chữ góc màn hình ở mọi vị trí
    wm_boxes = detect_static_corner_watermarks(source_video, canvas_w=1280, canvas_h=720)
    if wm_boxes:
        print(f"[*] [Renderer] Phát hiện {len(wm_boxes)} vùng Watermark/Logo ở góc: {wm_boxes} -> Đang kích hoạt delogo...")
        delogo_chain = ",".join([f"delogo=x={x}:y={y}:w={w}:h={h}:show=0" for (x, y, w, h) in wm_boxes])
        filter_parts.append(f"[vscaled]{delogo_chain}[vbase]")
    else:
        filter_parts.append("[vscaled]null[vbase]")

    last_v = "vbase"

    # Chỉ làm mờ sub tiếng Trung khi bật blur_zh=True (mặc định: False)
    if blur_zh:
        if zh_srt_path and Path(zh_srt_path).exists():
            blur_mask_ass = out_video.parent / f"temp_blur_mask_{out_video.stem}.ass"
            print("[*] [Renderer] Đang đo kích thước từng dòng sub để tạo Dynamic Blur Mask...")
            generate_dynamic_blur_ass(source_video, Path(zh_srt_path), blur_mask_ass)
            mask_rel = blur_mask_ass.name

            dur_val = duration if (duration and duration > 0) else 7200.0
            # Kỹ thuật Alpha-Masked Dynamic Blur trong FFmpeg:
            # - Tách vùng đáy (y=520..696) -> Làm mờ (boxblur=18:4, crop chiều cao chẵn 176px cho chuẩn yuv420p)
            # - Tạo một tấm nền đen cùng kích thước, vẽ hộp trắng từ blur_mask.ass lên làm Alpha Mask
            # - Dùng alphamerge để cắt vùng mờ đúng theo hình dáng hộp chữ của từng câu rồi đè lên video gốc!
            filter_parts.extend([
                "[vbase]split=2[vorig][vforblur]",
                "[vforblur]crop=1280:176:0:520,boxblur=18:4[vblurred_strip]",
                f"color=c=black:s=1280x720:r=30:d={dur_val:.1f},subtitles='{mask_rel}',crop=1280:176:0:520,format=gray[vmask]",
                "[vblurred_strip][vmask]alphamerge[vmasked_blur]",
                "[vorig][vmasked_blur]overlay=0:520:shortest=1[vclean]"
            ])
            last_v = "vclean"
        else:
            # Fallback nếu không truyền file SRT
            filter_parts.extend([
                "[vbase]crop=1280:150:0:535,boxblur=18:4[blurred_zh]",
                "[vbase][blurred_zh]overlay=0:535[vclean]"
            ])
            last_v = "vclean"

    if channel_icon and channel_icon.exists():
        filter_parts.append(f"[{last_v}][1:v]overlay=20:20[vbrand]")
        last_v = "vbrand"

    if not no_sub and ass_path and Path(ass_path).exists():
        try:
            sub_rel = os.path.relpath(ass_path, out_video.parent).replace("\\", "/")
        except Exception:
            sub_rel = Path(ass_path).name
        filter_parts.append(f"[{last_v}]subtitles='{sub_rel}'[vout]")
    else:
        filter_parts.append(f"[{last_v}]null[vout]")

    filter_v = ";".join(filter_parts)
    voice_idx = "2:a" if (channel_icon and channel_icon.exists()) else "1:a"

    # Best Practice Audio Mixing (SFX/BGM Preservation + Auto-Ducking):
    # 1. [voice_idx] được nhân đôi: 1 nhánh phát chính ([v_main]), 1 nhánh làm tín hiệu điều khiển ([v_sc]).
    # 2. [0:a] (âm thanh gốc chứa tiếng đánh nhau + thoại Trung) được lọc nhẹ dải 800Hz-2500Hz (-6dB) để làm dịu giọng Trung,
    #    sau đó đưa vào sidechaincompress: Hễ có giọng Việt ([v_sc]), âm thanh gốc tự động cụp xuống (ratio=10, threshold=0.015),
    #    khi hết câu tiếng Việt, âm thanh đánh nhau tự động bung lên lại 100% sau 300ms!
    filter_a = (
        f"[{voice_idx}]volume=1.65,asplit=2[v_main][v_sc];"
        f"[0:a]volume=0.95,equalizer=f=1400:width_type=h:width=1600:g=-5[bg_eq];"
        f"[bg_eq][v_sc]sidechaincompress=threshold=0.015:ratio=12:attack=15:release=320:makeup=1.0[bg_ducked];"
        f"[bg_ducked][v_main]amix=inputs=2:duration=first:dropout_transition=0,alimiter=limit=0.96[aout]"
    )

    inputs = ["-i", str(src_abs)]
    if channel_icon and Path(channel_icon).exists():
        inputs.extend(["-i", str(Path(channel_icon).resolve())])
    inputs.extend(["-i", str(voice_abs)])

    cmd_qsv = [
        FFMPEG, "-nostdin", "-y",
        *inputs,
        "-filter_complex", f"{filter_v};{filter_a}",
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_qsv",
        "-b:v", f"{bitrate_kbps}k",
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
    ]
    if duration and duration > 0:
        cmd_qsv.extend(["-t", str(duration)])
    cmd_qsv.append(str(temp_render))

    print(f"[*] [Renderer] Đang render video chuẩn Production với Dynamic Blur ({bitrate_kbps} kbps)...")
    t0 = time.time()
    qsv_ok = False
    try:
        res = subprocess.run(cmd_qsv, cwd=str(out_video.parent), capture_output=True, text=True)
        if res.returncode == 0 and temp_render.exists() and temp_render.stat().st_size > 1_000_000:
            # Kiểm tra thời lượng file output có đạt yêu cầu không
            actual_dur = get_media_duration(temp_render)
            expected_dur = duration if (duration and duration > 0) else get_media_duration(source_video)
            if actual_dur >= min(expected_dur * 0.90, expected_dur - 5.0):
                qsv_ok = True
                print(f"[OK] [Renderer] Render Intel Arc QSV thành công trong {(time.time()-t0):.1f}s (Thời lượng: {actual_dur:.1f}s)!")
            else:
                print(f"[!] [Renderer] QSV xuất file không đủ thời lượng ({actual_dur:.1f}s / {expected_dur:.1f}s), chuyển sang CPU...")
    except Exception as e:
        print(f"[!] [Renderer] Lỗi QSV: {e}")

    if not qsv_ok:
        print("[!] [Renderer] QSV fallback sang CPU libx264 CRF 18...")
        cmd_cpu = [
            FFMPEG, "-nostdin", "-y",
            *inputs,
            "-filter_complex", f"{filter_v};{filter_a}",
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
        if duration and duration > 0:
            cmd_cpu.extend(["-t", str(duration)])
        cmd_cpu.append(str(temp_render))
        subprocess.run(cmd_cpu, cwd=str(out_video.parent), check=True)
        print(f"[OK] [Renderer] Render CPU hoàn tất trong {(time.time()-t0):.1f}s!")

    if out_video.exists():
        try:
            out_video.unlink()
        except Exception:
            pass

    shutil.move(str(temp_render), str(out_video))

    mb = out_video.stat().st_size / (1024**2)
    print(f"🎉 [OK] [Renderer] Video thành phẩm xuất xưởng: {out_video.name} ({mb:.1f} MB)")
    return out_video
