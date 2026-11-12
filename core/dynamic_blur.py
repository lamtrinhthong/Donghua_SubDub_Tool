import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from core.common import format_ass_time
from core.srt_handler import parse_srt


def detect_sub_bbox_in_roi(
    cap: cv2.VideoCapture,
    time_sec: float,
    canvas_w: int = 1280,
    canvas_h: int = 720,
    roi_y1: int = 520,
    roi_y2: int = 695
) -> Optional[Tuple[int, int, int, int]]:
    """
    Nhảy tới đúng thời điểm giữa câu sub (time_sec), cắt vùng đáy video (ROI)
    và dùng bộ lọc đạo hàm ngang (Sobel X) + Morphology để bắt chính xác
    khung chữ nhật (x1, y1, x2, y2) ôm sát dòng phụ đề tiếng Trung.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    h_orig, w_orig = frame.shape[:2]
    if (w_orig, h_orig) != (canvas_w, canvas_h):
        frame = cv2.resize(frame, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)

    roi = frame[roi_y1:roi_y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Phụ đề phim hoạt hình luôn có độ tương phản viền rất cao (chữ sáng viền tối)
    # Dùng Black-hat & Top-hat kết hợp Sobel ngang để bắt nét chữ cực nhạy
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    # Chỉ giữ lại vùng có cả độ sáng cao (thân chữ) và cạnh sắc nét (viền chữ)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    _, edge_mask = cv2.threshold(abs_grad_x, 90, 255, cv2.THRESH_BINARY)

    # Nở nhẹ vùng sáng để giao với vùng cạnh
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.bitwise_and(cv2.dilate(bright_mask, k_small), cv2.dilate(edge_mask, k_small))

    # Nối liền các nét chữ Hán thành 1 khối dòng ngang (kernel 28x6)
    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 6))
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_line)
    morphed = cv2.erode(morphed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_boxes = []
    cx_center = canvas_w // 2
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Lọc các khối có chiều cao của dòng chữ (16px - 95px) và nằm gần trục giữa màn hình
        if 16 <= h <= 95 and w >= 24:
            box_cx = x + w // 2
            if abs(box_cx - cx_center) < 260:  # Sub luôn căn giữa màn hình
                valid_boxes.append((x, y, x + w, y + h))

    if not valid_boxes:
        return None

    x1 = min(b[0] for b in valid_boxes)
    y1 = min(b[1] for b in valid_boxes) + roi_y1
    x2 = max(b[2] for b in valid_boxes)
    y2 = max(b[3] for b in valid_boxes) + roi_y1

    # Căn đối xứng qua tâm ngang (vì sub Trung luôn căn giữa, đối xứng hóa giúp hộp cực đẹp)
    half_w = max(cx_center - x1, x2 - cx_center)
    half_w = min(560, max(45, half_w + 18))  # Cộng 18px đệm 2 bên
    x1_sym = cx_center - half_w
    x2_sym = cx_center + half_w

    y1_pad = max(roi_y1, y1 - 8)
    y2_pad = min(roi_y2, y2 + 8)
    return x1_sym, y1_pad, x2_sym, y2_pad


def generate_dynamic_blur_ass(
    source_video: Path,
    zh_srt_path: Path,
    out_blur_ass: Path,
    default_cy: int = 638,
    default_h: int = 52
) -> Path:
    """
    Sinh file ASS đóng vai trò là Dynamic Blur Mask:
    - Tự động đo kích thước dòng sub tiếng Trung của từng câu (kết hợp OpenCV + độ dài chữ Hán).
    - Vẽ hộp bo tròn viền mềm (\blur6) đúng bằng kích thước dòng sub tại đúng thời điểm câu đó xuất hiện!
    """
    subs = parse_srt(zh_srt_path)
    cap = cv2.VideoCapture(str(source_video))
    has_cv = cap.isOpened()

    ass_header = """[Script Info]
Title: Dynamic Subtitle Blur Mask
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BlurMask,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    r = 12  # Bo góc mềm mại cho vùng làm mờ
    k = int(r * 0.55228475)
    total_subs = len(subs)

    try:
        for idx_s, s in enumerate(subs):
            if (idx_s + 1) % 50 == 0 or (idx_s + 1) == total_subs:
                print(f"    - Quét mask Dynamic Blur: {idx_s + 1}/{total_subs} câu...")
            s_val = max(0.0, s["start"] - 0.04)  # Mở mask sớm 40ms để che kín ngay frame đầu
            e_val = s["end"] + 0.04              # Đóng mask trễ 40ms để không hở chữ cuối
            t_mid = (s["start"] + s["end"]) / 2.0

            bbox = None
            if has_cv:
                bbox = detect_sub_bbox_in_roi(cap, t_mid)

            # Kết hợp với ước lượng độ dài ký tự tiếng Trung (mỗi chữ Hán ở 720p rộng ~34px)
            clean_text = s["text"].replace("\n", "").strip()
            char_count = max(2, len(clean_text))
            est_half_w = min(540, max(50, int(char_count * 34) // 2 + 22))

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Kẹp thêm với ước lượng ký tự để không bị lọt chữ ở rìa nếu chữ ngoài cùng hơi mờ
                cx = 640
                detected_hw = (x2 - x1) // 2
                final_hw = max(detected_hw, int(est_half_w * 0.85))
                x1, x2 = cx - final_hw, cx + final_hw
            else:
                # Fallback chuẩn xác theo số lượng chữ Hán của câu đó
                cx = 640
                x1, x2 = cx - est_half_w, cx + est_half_w
                y1, y2 = default_cy - default_h // 2, default_cy + default_h // 2

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

            s_str = format_ass_time(s_val)
            e_str = format_ass_time(e_val)
            # Dùng \blur6 để mép của vùng làm mờ hòa quyện tự nhiên vào nền video, không bị gắt cạnh
            events.append(
                f"Dialogue: 0,{s_str},{e_str},BlurMask,,0,0,0,,"
                f"{{\\pos(0,0)\\p1\\1c&HFFFFFF&\\bord0\\shad0\\blur6}}{path}{{\\p0}}\n"
            )
    finally:
        if has_cv:
            cap.release()

    out_blur_ass.parent.mkdir(parents=True, exist_ok=True)
    with open(out_blur_ass, "w", encoding="utf-8") as f:
        f.write(ass_header + "".join(events))

    return out_blur_ass


def detect_static_corner_watermarks(
    video_path: Path,
    canvas_w: int = 1280,
    canvas_h: int = 720,
    num_samples: int = 14
) -> List[Tuple[int, int, int, int]]:
    """
    Tự động dò tìm mọi Watermark / Logo Bilibili / Chữ cảnh báo ở cả 4 góc màn hình:
    - Phân tích sự bất biến của đạo hàm cạnh (Temporal Edge Persistence) qua `num_samples` frame.
    - Bắt chính xác tọa độ (x, y, w, h) dù mỗi video đặt logo ở góc khác nhau hay tên kênh dài ngắn khác nhau.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 30:
        cap.release()
        return []

    # Lấy mẫu các frame từ 8% đến 92% thời lượng video để tránh intro/outro đen
    sample_indices = np.linspace(int(total_frames * 0.08), int(total_frames * 0.92), num_samples, dtype=int)
    edge_frames = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if (frame.shape[1], frame.shape[0]) != (canvas_w, canvas_h):
            frame = cv2.resize(frame, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Tính cường độ cạnh (Gradient Magnitude) để bắt viền chữ & logo bán trong suốt
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edge_frames.append(mag)

    cap.release()
    if len(edge_frames) < 4:
        return []

    # [CHÌA KHÓA]: Lấy phân vị 25% theo trục thời gian (axis=0).
    # Cảnh phim chuyển động sẽ bị triệt tiêu về 0, chỉ có Logo/Watermark đứng yên mới giữ nguyên cường độ cạnh!
    persistent_edges = np.percentile(np.stack(edge_frames, axis=0), 25, axis=0)
    persistent_u8 = np.clip(persistent_edges, 0, 255).astype(np.uint8)

    # Chỉ cho phép tìm trong 4 vùng góc (Corner Zones) để không bao giờ chạm vào nhân vật ở giữa màn hình
    corner_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cw, ch = int(canvas_w * 0.32), int(canvas_h * 0.24)  # Mỗi góc rộng 32% màn hình, cao 24% màn hình
    corner_mask[0:ch, 0:cw] = 255                        # Góc trên trái (Top-Left)
    corner_mask[0:ch, canvas_w - cw:canvas_w] = 255      # Góc trên phải (Top-Right)
    corner_mask[canvas_h - ch:canvas_h, 0:cw] = 255      # Góc dưới trái (Bottom-Left)
    corner_mask[canvas_h - ch:canvas_h, canvas_w - cw:canvas_w] = 255  # Góc dưới phải (Bottom-Right)

    masked_edges = cv2.bitwise_and(persistent_u8, corner_mask)
    _, thresh = cv2.threshold(masked_edges, 28, 255, cv2.THRESH_BINARY)

    # Nối liền cụm icon TV Bilibili + chữ tên kênh + chữ bilibili thành 1 khối duy nhất
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 12))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Bỏ qua các chấm nhiễu quá nhỏ
        if w < 25 or h < 14:
            continue
        # Mở rộng nhẹ 6px mỗi chiều và kẹp trong phạm vi an toàn của filter delogo (cách mép màn hình ít nhất 2px)
        x1 = max(2, x - 6)
        y1 = max(2, y - 6)
        x2 = min(canvas_w - 3, x + w + 6)
        y2 = min(canvas_h - 3, y + h + 6)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes
