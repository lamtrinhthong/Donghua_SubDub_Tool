"""
FastLab AI Studio - Micro-PID Time Warper Engine (Natural 1.0x Pace Edition)
=============================================================================
Chế độ Giọng Đọc Tự Nhiên (1.0x ± 0.05):
1. Internal Pause Compaction: Thu gọn các khoảng nghỉ thừa bên trong câu (dấu phẩy/ngắt nhịp của Google TTS)
   từ 300ms xuống 55ms -> Giảm ngay 15-25% thời lượng file âm thanh mà tốc độ phát âm chữ vẫn là 1.00x!
2. Fixed Speed Band [1.00x - 1.05x]: Giữ giọng đọc đều đặn, tự nhiên tuyệt đối giữa mọi câu.
   (Trước đây code clip tốc độ vào [1.00 - 1.15] và không dùng min_speed -> lệch với mô tả.)
3. Elastic Gap Borrowing & Self-Healing: Tận dụng tối đa khoảng lặng giữa các câu để không cần tua nhanh.
4. Hard Non-Overlap Guarantee: Hai câu KHÔNG BAO GIỜ chồng lấn nhau. Khi không thể vừa khung thời gian,
   phần thừa sẽ bị cắt bỏ (được đếm và báo cáo) thay vì cộng dồn âm thanh (bug "master_audio += audio").
5. Micro-PID Drift Controller: PID thật sự (kp/ki/kd) điều chỉnh tốc độ trong dải hẹp để kéo lại
   độ trôi tích lũy của timeline, có chống windup.
"""

import math
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from core.common import FFMPEG


def _atempo_chain(speed: float) -> str:
    """
    Tạo chuỗi filter `atempo` hợp lệ.
    FFmpeg chỉ chấp nhận atempo trong khoảng [0.5, 2.0], nên tốc độ ngoài khoảng này
    phải được chia thành nhiều tầng.
    """
    factors: List[float] = []
    s = float(speed)
    if s <= 0:
        s = 1.0
    while s > 2.0:
        factors.append(2.0)
        s /= 2.0
    while s < 0.5:
        factors.append(0.5)
        s /= 0.5
    factors.append(s)
    return ",".join(f"atempo={f:.4f}" for f in factors)


def _estimate_noise_floor(audio: np.ndarray, sr: int = 48000, win_ms: float = 30.0) -> float:
    """
    Ước lượng mức nhiễu nền của tín hiệu (mức RMS thấp nhất trong các cửa sổ ngắn).
    Dùng để đặt ngưỡng "có tiếng nói" một cách ổn định, không phụ thuộc vào độ lớn tổng thể.
    """
    if len(audio) < 1024:
        return 0.0
    win = max(1, int(sr * win_ms / 1000.0))
    n_win = len(audio) // win
    if n_win < 4:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    trimmed = audio[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    # Lấy 10% cửa sổ yếu nhất làm nhiễu nền
    k = max(1, int(len(rms) * 0.10))
    return float(np.sort(rms)[:k].mean())


def trim_and_compact_speech(
    audio_data: np.ndarray,
    sr: int,
    pad_ms: int = 20,
    max_internal_pause_ms: int = 55
) -> np.ndarray:
    """
    1. Cắt bỏ hoàn toàn khoảng lặng chết ở đầu và cuối câu (Dead Silence Trimming).
    2. Thu gọn các khoảng nghỉ quá dài BÊN TRONG câu (do dấu phẩy/ngắt nhịp của TTS)
       xuống tối đa `max_internal_pause_ms` (55ms).
    -> Giúp câu thoại gọn hơn 15-25% thời gian mà tốc độ phát âm chữ vẫn giữ nguyên 1.00x!

    Sửa lỗi (so với bản cũ):
      - `np.convolve(..., mode="same")` làm biên độ ở hai đầu bị chia nhỏ hơn thực tế,
        dẫn tới việc tín hiệu toàn đầu/cuối bị coi nhầm là im lặng và cắt mất âm đầu.
        Nay tính envelope trên tín hiệu đã pad bằng giá trị biên.
      - Ngưỡng "có tiếng" nay dùng max(2.2% peak, 2.5 * nhiễu nền) -> ổn định hơn với âm lượng thấp.
    """
    if len(audio_data) == 0:
        return audio_data
    audio = np.asarray(audio_data, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-5:
        return audio

    # Tính biên độ mượt theo cửa sổ 10ms để nhận diện chính xác khoảng lặng
    win_size = max(1, int(sr * 0.010))
    abs_data = np.abs(audio)
    kernel = np.ones(win_size) / win_size

    # Pad bằng giá trị biên để convolution 'same' không bị lệch ở hai đầu
    pad = win_size
    padded = np.pad(abs_data, (pad, pad), mode="edge")
    envelope = np.convolve(padded, kernel, mode="same")[pad:pad + len(abs_data)]

    noise_floor = _estimate_noise_floor(audio, sr=sr)
    thresh = max(0.022 * peak, 2.5 * noise_floor)
    is_speech = envelope > thresh
    speech_indices = np.where(is_speech)[0]
    if len(speech_indices) == 0:
        return audio

    # 1. Cắt đầu và đuôi
    pad_samples = int(sr * (pad_ms / 1000.0))
    start_idx = max(0, int(speech_indices[0]) - pad_samples)
    end_idx = min(len(audio), int(speech_indices[-1]) + pad_samples)
    trimmed = audio[start_idx:end_idx]
    mask = is_speech[start_idx:end_idx]

    # 2. Nén khoảng lặng nội câu (Internal Pause Compaction)
    max_pause_samples = int(sr * (max_internal_pause_ms / 1000.0))
    chunks = []
    n = len(trimmed)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            chunks.append(trimmed[i:j])
            i = j
        else:
            j = i
            while j < n and not mask[j]:
                j += 1
            silence_len = j - i
            if silence_len > max_pause_samples:
                # Giữ lại đúng max_pause_samples và nối mượt
                half = max_pause_samples // 2
                compact_sil = np.concatenate([trimmed[i:i + half], trimmed[j - half:j]])
                chunks.append(compact_sil)
            else:
                chunks.append(trimmed[i:j])
            i = j

    compacted = np.concatenate(chunks) if chunks else trimmed

    # Micro fade-in / fade-out 6ms chống tiếng bụp (click)
    fade_len = min(int(sr * 0.006), len(compacted) // 4)
    if fade_len > 1:
        compacted = compacted.copy()
        compacted[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=compacted.dtype)
        compacted[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=compacted.dtype)

    return compacted


def normalize_master(audio: np.ndarray, target_rms_dbfs: float = -20.0,
                     peak_ceiling: float = 0.95, sr: int = 48000) -> np.ndarray:
    """
    Chuẩn hóa âm lượng master track theo RMS của các đoạn CÓ TIẾNG (ổn định độ lớn
    cảm nhận) thay vì theo peak (dễ bị "bơm" âm lượng và méo khi có một đoạn đột ngột lớn).

    Vì master track chứa rất nhiều khoảng lặng, RMS toàn cục sẽ thấp hơn nhiều so với
    mức tiếng thực -> dùng phân vị 95 của RMS theo cửa sổ 100ms để đo đúng "mức tiếng".

    - Gain chính đến từ RMS mục tiêu.
    - Sau đó peak ceiling chỉ đóng vai trò trần an toàn (scale đồng nhất, không bóp méo).
    """
    if len(audio) == 0:
        return audio
    out = np.asarray(audio, dtype=np.float32)

    win = max(1, int(sr * 0.100))
    n_win = len(out) // win
    if n_win < 4:
        win = max(1, len(out) // 8)
        n_win = len(out) // win
    if n_win < 4:
        return out

    frames = out[: n_win * win].reshape(n_win, win).astype(np.float64)
    win_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Đo mức tiếng thực: chọn các cửa sổ có năng lượng đáng kể, nối thành vùng liên tục,
    # rồi tính RMS trực tiếp trên các mẫu thuộc vùng đó.
    # (Dùng mean của window RMS sẽ lệch với tín hiệu nhiễu do phân bố RMS cửa sổ rất rộng.)
    peak_rms = float(np.max(win_rms))
    if peak_rms <= 1e-6:
        return out
    thr = 0.30 * peak_rms
    active_mask = win_rms >= thr
    if not np.any(active_mask):
        return out
    # Mở rộng mask cửa sổ về đúng độ dài tín hiệu (phần đuôi bị cắt bởi reshape)
    sample_mask = np.zeros(len(out), dtype=bool)
    covered = n_win * win
    sample_mask[:covered] = np.repeat(active_mask, win)[:covered]
    # Phần đuôi nằm trong cửa sổ cuối cùng -> gán theo trạng thái cửa sổ đó
    if covered < len(out) and active_mask[-1]:
        sample_mask[covered:] = True
    speech_samples = out[sample_mask]
    if len(speech_samples) < 1024:
        return out
    speech_rms = float(np.sqrt(np.mean(speech_samples.astype(np.float64) ** 2)))
    if speech_rms <= 1e-6:
        return out

    target_rms = 10 ** (target_rms_dbfs / 20.0)
    gain = float(np.clip(target_rms / speech_rms, 0.25, 16.0))
    out = out * gain
    peak = float(np.max(np.abs(out)))
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out


class MicroPIDTimeWarper:
    """
    Bộ điều khiển thời gian cho track thuyết minh.

    Nguyên tắc bất di bất dịch:
        - `actual_start` của câu luôn >= `actual_end` của câu trước + `min_breath_pause`.
          -> KHÔNG BAO GIỜ có 2 câu chồng lấn nhau.
        - Khi audio vẫn dài hơn khung thời gian cho phép, phần thừa bị CẮT BỎ (đếm `trimmed_sentences`),
          thay vì cộng dồn (bug cũ: `master_audio[start:end] += audio_chunk`).
        - Subtitle hiển thị theo `sub_end` = thời điểm audio thực sự kết thúc.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        kp: float = 0.25,
        ki: float = 0.05,
        kd: float = 0.02,
        min_speed: float = 1.00,        # Không bao giờ chậm hơn 1.00x (giọng đọc tự nhiên)
        max_speed: float = 1.05,        # Dải tốc độ chuẩn [1.00x - 1.05x]
        emergency_speed: float = 1.20,  # Ngưỡng cứu hộ (+20%) nếu dồn quá nhiều câu sát nhau
        hard_cap_speed: float = 1.40,   # Trần cứng: ưu tiên giữ đủ lời thoại thay vì cắt bỏ
        min_breath_pause: float = 0.025,
        lead_in: float = 0.060,         # Cho phép câu bắt đầu sớm tối đa 60ms
        drift_penalty: float = 0.35,    # Ngưỡng trôi (giây) để kích hoạt emergency speed
        integral_limit: float = 2.0,    # Chống windup của PID
        target_rms_dbfs: float = -20.0
    ):
        self.sr = sample_rate
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.emergency_speed = emergency_speed
        self.hard_cap_speed = hard_cap_speed
        self.min_breath = min_breath_pause
        self.lead_in = lead_in
        self.drift_penalty = drift_penalty
        self.integral_limit = integral_limit
        self.target_rms_dbfs = target_rms_dbfs

    # ------------------------------------------------------------------
    # 1. Rời rạc hóa timeline
    # ------------------------------------------------------------------
    def differentiate_timeline(self, events: List[Dict], total_duration: float = 600.0) -> List[Dict]:
        """
        Rời rạc hóa dòng thời gian và tính toán quỹ thời gian của từng câu + các câu lân cận.
        """
        diff_events = []
        n = len(events)
        for i, ev in enumerate(events):
            t_start = float(ev["start"])
            t_end = float(ev["end"])
            t_dur = max(0.08, t_end - t_start)

            if i + 1 < n:
                next_start = float(events[i + 1]["start"])
                avail_window = max(0.08, next_start - t_start - self.min_breath)
            else:
                next_start = total_duration
                avail_window = max(0.15, total_duration - t_start)

            text = ev.get("text_vi", "").strip()
            words = text.split()
            syllable_count = len(words)

            diff_item = {
                "id": ev.get("id", i + 1),
                "target_start": t_start,
                "target_end": t_end,
                "target_dur": t_dur,
                "next_target_start": next_start,
                "avail_window": avail_window,
                "text_vi": text,
                "text_zh": ev.get("text_zh", ""),
                "syllables": syllable_count,
                "v_target": round(syllable_count / t_dur, 2) if t_dur > 0 else 3.2
            }
            for k, v in ev.items():
                if k not in diff_item:
                    diff_item[k] = v
            diff_events.append(diff_item)
        return diff_events

    # ------------------------------------------------------------------
    # 2. Tổng hợp master track
    # ------------------------------------------------------------------
    def process_and_synthesize(
        self,
        diff_events: List[Dict],
        tts_cache_dir: Path,
        total_duration: float,
        out_audio_path: Path
    ) -> Tuple[Path, List[Dict], Dict]:
        """
        Tổng hợp Master Track với tốc độ khóa chặt quanh 1.0x (±0.05).
        """
        tts_cache_dir = Path(tts_cache_dir)
        tts_cache_dir.mkdir(parents=True, exist_ok=True)
        total_samples = int(np.ceil(total_duration * self.sr))
        master_audio = np.zeros(total_samples, dtype=np.float32)

        scheduled_events: List[Dict] = []
        drift_records: List[float] = []
        speed_records: List[float] = []
        collisions = 0
        trimmed_sentences = 0
        delayed_sentences = 0
        total_trimmed_sec = 0.0
        max_overlap_ms = 0.0

        # PID state
        integral = 0.0
        prev_error = 0.0
        # "Nợ thời gian": tổng độ trễ tích luỹ của các câu trước.
        # Khi nợ > 0, các câu sau được phép chạy nhanh hơn (emergency) để kéo lại timeline.
        time_debt = 0.0

        prev_end = 0.0  # actual_end của câu trước (đảm bảo không chồng lấn)

        for idx, ev in enumerate(diff_events):
            t_target_start = ev["target_start"]
            t_target_end = ev["target_end"]
            next_target_start = ev["next_target_start"]
            sent_id = ev["id"]

            if t_target_start >= total_duration:
                break

            wav_file = tts_cache_dir / f"sent_{sent_id:04d}.wav"
            if not wav_file.exists():
                continue

            data_raw, sr_raw = sf.read(str(wav_file))
            if data_raw.ndim > 1:
                data_raw = data_raw.mean(axis=1)

            # 1. Cắt khoảng lặng đầu/cuối + Thu gọn khoảng nghỉ dấu phẩy bên trong câu
            data_clean = trim_and_compact_speech(
                data_raw, self.sr, pad_ms=18, max_internal_pause_ms=55
            )
            raw_dur = len(data_clean) / self.sr
            if raw_dur <= 0.01:
                continue

            # --------------------------------------------------------------
            # 2. Chọn điểm bắt đầu thực tế (hard non-overlap)
            # --------------------------------------------------------------
            earliest_allowed = t_target_start - self.lead_in if idx > 0 else t_target_start
            if scheduled_events:
                earliest_allowed = max(earliest_allowed, prev_end + self.min_breath)

            # KHÔNG giới hạn trên bởi `latest_start`: nếu câu trước đã trễ thì câu này
            # buộc phải trễ theo (để đảm bảo KHÔNG BAO GIỜ chồng lấn).
            # Phần âm dư sẽ bị cắt cứng ở bước 7 (ưu tiên giữ đủ lời thoại).
            actual_start = max(t_target_start, earliest_allowed)
            actual_start = max(actual_start, 0.0)

            # Nếu bị đẩy lùi khỏi vị trí gốc -> ghi nhận va chạm (delay)
            delayed = actual_start - t_target_start > self.min_breath + 0.005
            if delayed:
                delayed_sentences += 1
                max_overlap_ms = max(max_overlap_ms, (actual_start - t_target_start) * 1000.0)
            # --------------------------------------------------------------
            # 3. Khung thời gian khả dụng cho audio của câu này
            # --------------------------------------------------------------
            real_avail_win = max(0.05, next_target_start - actual_start - self.min_breath)
            real_avail_win = min(real_avail_win, max(0.05, total_duration - actual_start))

            # --------------------------------------------------------------
            # 4. Micro-PID: điều chỉnh theo độ trôi tích lũy
            # --------------------------------------------------------------
            error = actual_start - t_target_start          # >0 = đang trễ
            integral = float(np.clip(integral + error, -self.integral_limit, self.integral_limit))
            derivative = error - prev_error
            prev_error = error
            pid_corr = (self.kp * error + self.ki * integral + self.kd * derivative)

            # --------------------------------------------------------------
            # 5. Quyết định tốc độ
            # --------------------------------------------------------------
            needed_speed = raw_dur / real_avail_win
            # Ngưỡng cứu hộ: dùng `emergency_speed` khi
            #   (a) nội dung dài hơn khung thời gian (needed_speed > max_speed), hoặc
            #   (b) câu đang trễ so với timeline gốc, hoặc
            #   (c) đang có "nợ thời gian" tích luỹ từ các câu trước.
            needs_rescue = (
                needed_speed > self.max_speed
                or error > self.drift_penalty
                or time_debt > self.drift_penalty
            )
            # Ưu tiên giữ đủ lời thoại: nếu chỉ cần tăng tốc nhẹ để vừa khung thì
            # dùng emergency_speed, chỉ khi vượt quá mới dùng hard cap.
            if needed_speed > self.emergency_speed:
                upper_limit = self.hard_cap_speed
            elif needs_rescue:
                upper_limit = self.emergency_speed
            else:
                upper_limit = self.max_speed
            # PID chỉ được phép "đẩy nhanh" (không được làm chậm dưới 1.0x)
            pid_speed = 1.0 + max(0.0, pid_corr * 0.10)
            speed = float(np.clip(max(needed_speed, pid_speed), self.min_speed, upper_limit))
            speed = float(np.clip(speed, self.min_speed, max(self.max_speed, upper_limit)))

            speed_records.append(speed)

            # --------------------------------------------------------------
            # 6. Áp dụng tốc độ bằng FFmpeg atempo
            # --------------------------------------------------------------
            if abs(speed - 1.0) > 0.005:
                trimmed_wav = tts_cache_dir / f"sent_{sent_id:04d}_trimmed.wav"
                sped_wav = tts_cache_dir / f"sent_{sent_id:04d}_pid_sped.wav"
                sf.write(str(trimmed_wav), data_clean, self.sr)
                cmd = [
                    FFMPEG, "-nostdin", "-y",
                    "-i", str(trimmed_wav),
                    "-filter:a", _atempo_chain(speed),
                    "-ar", str(self.sr), "-ac", "1",
                    str(sped_wav)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if sped_wav.exists():
                    audio_chunk, _ = sf.read(str(sped_wav))
                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.mean(axis=1)
                else:
                    audio_chunk = data_clean
            else:
                audio_chunk = data_clean

            # --------------------------------------------------------------
            # 7. Cắt cứng nếu vẫn vượt khung (không bao giờ chồng lấn câu sau)
            # --------------------------------------------------------------
            max_samples = int(round(real_avail_win * self.sr))
            trimmed_ms = 0.0
            if len(audio_chunk) > max_samples:
                excess = len(audio_chunk) - max_samples
                trimmed_ms = excess / self.sr * 1000.0
                total_trimmed_sec += excess / self.sr
                trimmed_sentences += 1
                collisions += 1  # Audio tràn sang vùng của câu kế tiếp -> phải cắt
                audio_chunk = audio_chunk[:max_samples]

            final_dur = len(audio_chunk) / self.sr
            actual_end = min(total_duration, actual_start + final_dur)

            # --------------------------------------------------------------
            # 8. Chèn vào Master Track (KHÔNG cộng dồn chồng lấn)
            # --------------------------------------------------------------
            start_samp = int(round(actual_start * self.sr))
            end_samp = min(total_samples, start_samp + len(audio_chunk))
            insert_len = end_samp - start_samp

            if insert_len > 0:
                master_audio[start_samp:end_samp] = audio_chunk[:insert_len]

            prev_end = actual_end
            start_drift = abs(actual_start - t_target_start)
            drift_records.append(start_drift)
            # Cập nhật nợ thời gian: câu bị cắt cứng tạo ra "nợ" cho các câu sau.
            time_debt += trimmed_ms / 1000.0

            # --------------------------------------------------------------
            # 9. Thời gian hiển thị phụ đề: bám theo audio thực tế
            # --------------------------------------------------------------
            sub_end = max(actual_end, actual_start + 0.05)
            sub_end = min(sub_end, total_duration)

            sched_item = {
                "id": sent_id,
                "start": round(actual_start, 3),
                "end": round(sub_end, 3),
                "target_start": t_target_start,
                "target_end": t_target_end,
                "actual_start": round(actual_start, 3),
                "actual_end": round(actual_end, 3),
                "sub_end": round(sub_end, 3),
                "actual_dur": round(final_dur, 3),
                "raw_dur": round(raw_dur, 3),
                "speed_applied": round(speed, 3),
                "drift_ms": round(start_drift * 1000.0, 2),
                "lag_ms": round((actual_start - t_target_start) * 1000.0, 2),
                "trimmed_ms": round(trimmed_ms, 2),
                "text_vi": ev["text_vi"],
                "text_zh": ev.get("text_zh", "")
            }
            for k, v in ev.items():
                if k not in sched_item:
                    sched_item[k] = v
            scheduled_events.append(sched_item)

        # Chuẩn hóa âm lượng đầu ra theo RMS (ổn định hơn peak-only)
        master_audio = normalize_master(master_audio, target_rms_dbfs=self.target_rms_dbfs, sr=self.sr)

        sf.write(str(out_audio_path), master_audio, self.sr)

        n_sent = len(scheduled_events)
        metrics = {
            "total_sentences": n_sent,
            "collisions": collisions,
            "delayed_sentences": delayed_sentences,
            "trimmed_sentences": trimmed_sentences,
            "total_trimmed_ms": round(total_trimmed_sec * 1000.0, 2),
            "max_overlap_ms": round(max_overlap_ms, 2),
            "max_drift_ms": round(max(drift_records) * 1000.0, 2) if drift_records else 0.0,
            "mean_drift_ms": round(float(np.mean(drift_records)) * 1000.0, 2) if drift_records else 0.0,
            "mean_speed": round(float(np.mean(speed_records)), 3) if speed_records else 1.0,
            "max_speed_used": round(float(np.max(speed_records)), 3) if speed_records else 1.0,
            "zero_drift_rate": f"{round(sum(1 for d in drift_records if d < 0.080) / max(1, n_sent) * 100, 1)}% (1.0x Natural Sync)"
        }

        return out_audio_path, scheduled_events, metrics
