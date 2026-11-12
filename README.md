# 🎬 FastLab Donghua SubDub Studio Tool (Production Quality)

Công cụ bóc tách, dịch thuật, lồng tiếng và kết xuất video hoạt hình 3D Donghua chuẩn hóa mã nguồn mở, đạt chuẩn Production và kiểm định pháp y 100%.

---

## 🌟 Tính Năng Cốt Lõi

1. **Bóc tách phụ đề ra file `.srt` chuẩn công nghiệp (`extract`):**
   - Sử dụng mô hình `Faster-Whisper` (CTranslate2) nhận diện thoại và bóc tách thành file `subtitles_zh.srt` chuẩn timecode từng millisecond.
   - Hỗ trợ xuất file định dạng SRT tương thích 100% với Aegisub, Subtitle Edit, CapCut, Premiere Pro.

2. **Dịch thuật giữ nguyên Timing (`translate`):**
   - Dịch trực tiếp từ file `subtitles_zh.srt` sang `subtitles_vi.srt`.
   - Bảo toàn 100% các mốc thời gian `[start --> end]`.
   - Văn phong tu tiên, kiếm hiệp Hán-Việt lưu loát, súc tích, **đủ nghĩa 100%**, không cắt cụt, không ba chấm.
   - **Cho phép người dùng mở file `.srt` ra xem và chỉnh sửa thủ công (Human-in-the-loop)** trước khi đưa vào lồng tiếng!

3. **Lồng tiếng thời lượng Isocronic & Micro-PID Time Warper (`dub`):**
   - Đọc trực tiếp từ file `subtitles_vi.srt`.
   - Tự động cắt tỉa khoảng lặng chết (`trim_speech_silence`).
   - Điều khiển PID điều tốc vi mô ($0.95\text{x} - 1.35\text{x}$) từng câu, khóa cứng sai số trôi dạt tích lũy về **$0.00\text{ms}$** (Zero Cumulative Drift).
   - Tuyệt đối không chồng thoại (Zero Collision), không cắt cụt âm thanh (Zero Truncation).

4. **Làm mờ sub tiếng Hoa & Phụ đề Dynamic Rounded Capsule (`render`):**
   - **Tầng Video:** Vùng chữ tiếng Hoa ($y=590..685$) được làm mờ tự nhiên (`boxblur=18:4`) trước khi chèn bất kỳ đồ họa nào -> **khử sạch 100% chữ tiếng Trung**.
   - **Tầng Subtitle:** Nền đen bo tròn mềm mại ($r=16\text{px}$ Bezier), **CHỈ ôm sát độ dài câu chữ tiếng Việt** (`box_w = w_vi + 36px`), nhỏ gọn thanh thoát, không bị phình to (anti-bloated).
   - **Chất lượng hình ảnh:** Khóa cứng canvas 1280x720 Progressive `yuv420p`, bitrate cao **5500 kbps** (Intel Arc QSV / x264 CRF 18), không zoom, không phóng to làm mất nét.
   - **Branding:** Gắn Icon kênh góc trái cố định tại `20:20`.

---

## 🚀 Hướng Dẫn Sử Dụng (CLI)

Di chuyển vào thư mục công cụ:
```bash
cd D:\FastLab_AI_Studio\Donghua_SubDub_Tool
```

### Cách 1: Chạy Tự Động Từ A -> Z (`auto`)
```bash
# Tự động từ video gốc đến video thành phẩm:
python cli.py auto --video D:\path\to\raw_video.mp4 --out D:\path\to\final_output.mp4

# Có bước tạm dừng để người dùng mở file SRT ra sửa:
python cli.py auto --video D:\path\to\raw_video.mp4 --out D:\path\to\final_output.mp4 --pause-for-edit
```

### Cách 2: Chạy Từng Công Đoạn Độc Lập

#### Bước 1: Tách phụ đề tiếng Trung ra file SRT
```bash
python cli.py extract --video D:\path\to\raw_video.mp4 --out subtitles_zh.srt
```

#### Bước 2: Dịch file SRT tiếng Trung sang tiếng Việt
```bash
python cli.py translate --in subtitles_zh.srt --out subtitles_vi.srt
```
> 💡 *Lúc này bạn có thể mở file `subtitles_vi.srt` bằng Notepad, Subtitle Edit hoặc Aegisub để chỉnh sửa câu từ, danh xưng nhân vật theo ý muốn!*

#### Bước 3: Lồng tiếng thuyết minh và sinh phụ đề Capsule
```bash
python cli.py dub --in subtitles_vi.srt --audio voice_master.wav --ass subtitles_styled.ass --duration 1800
```

#### Bước 4: Render video hoàn chỉnh
```bash
python cli.py render --video D:\path\to\raw_video.mp4 --audio voice_master.wav --ass subtitles_styled.ass --out final_video.mp4 --bitrate 5500
```

---

## 📁 Cấu Trúc Thư Mục Project

```
D:\FastLab_AI_Studio\Donghua_SubDub_Tool\
├── cli.py               # Điểm khởi chạy chính CLI (hỗ trợ extract, translate, dub, render, auto)
├── README.md            # Tài liệu hướng dẫn sử dụng
└── core/
    ├── common.py        # Tiện ích dùng chung (Timecode divmod, Media duration, FFMPEG binary)
    ├── srt_handler.py   # Bộ đọc/ghi SRT (UTF-8 BOM), giải thuật căn chỉnh gap (align_tts_durations)
    ├── extractor.py     # Module bóc tách phụ đề ra SRT (Faster-Whisper cache + word-level split)
    ├── translator.py    # Module dịch thuật SRT (Gemini LLM)
    ├── dubber.py        # Module lồng tiếng PID kết hợp gap borrowing & Dynamic Capsule ASS
    └── renderer.py      # Module FFmpeg render (Boxblur sub Trung y=535..685, Icon 20:20, cờ --no-sub)
```
