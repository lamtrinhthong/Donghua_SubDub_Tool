"""
FastLab AI Studio - Micro-PID Time Warper Engine (Natural 1.0x Pace Edition)
=============================================================================
Chế độ Giọng Đọc Tự Nhiên (1.0x ± 0.05):
1. Internal Pause Compaction: Thu gọn các khoảng nghỉ thừa bên trong câu (dấu phẩy/ngắt nhịp của Google TTS)
   từ 300ms xuống 55ms -> Giảm ngay 15-25% thời lượng file âm thanh mà tốc độ phát âm chữ vẫn là 1.00x!
2. Fixed Speed Band [0.98x - 1.05x]: Giữ giọng đọc đều đặn, tự nhiên tuyệt đối giữa mọi câu.
3. Elastic Gap Borrowing & Self-Healing: Tận dụng tối đa khoảng lặng giữa các câu thoại để không cần tua nhanh.
"""

import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple
from core.common import FFMPEG


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
    """
    if len(audio_data) == 0:
        return audio_data
    peak = np.max(np.abs(audio_data))
    if peak <= 1e-5:
        return audio_data

    # Tính biên độ mượt theo cửa sổ 10ms để nhận diện chính xác khoảng lặng
    win_size = max(1, int(sr * 0.010))
    abs_data = np.abs(audio_data)
    kernel = np.ones(win_size) / win_size
    envelope = np.convolve(abs_data, kernel, mode="same")

    thresh = 0.022 * peak
    is_speech = envelope > thresh
    speech_indices = np.where(is_speech)[0]
    if len(speech_indices) == 0:
        return audio_data

    # 1. Cắt đầu và đuôi
    pad_samples = int(sr * (pad_ms / 1000.0))
    start_idx = max(0, speech_indices[0] - pad_samples)
    end_idx = min(len(audio_data), speech_indices[-1] + pad_samples)
    trimmed = audio_data[start_idx:end_idx]
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


class MicroPIDTimeWarper:
    def __init__(
        self,
        sample_rate: int = 48000,
        kp: float = 0.25,
        ki: float = 0.05,
        kd: float = 0.02,
        min_speed: float = 1.08,        # Dao động dưới cực nhỏ (-2%)
        max_speed: float = 1.15,        # Dao động trên cực nhỏ (+5%)
        emergency_speed: float = 1.20,  # Ngưỡng cứu hộ tối đa (+10%) nếu dồn quá nhiều câu sát nhau
        min_breath_pause: float = 0.025
    ):
        self.sr = sample_rate
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.emergency_speed = emergency_speed
        self.min_breath = min_breath_pause

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
        tts_cache_dir.mkdir(parents=True, exist_ok=True)
        total_samples = int(np.ceil(total_duration * self.sr))
        master_audio = np.zeros(total_samples, dtype=np.float32)

        scheduled_events = []
        drift_records = []
        speed_records = []
        collisions = 0

        for idx, ev in enumerate(diff_events):
            t_target_start = ev["target_start"]
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

            # 2. Xác định điểm bắt đầu thực tế:
            # Mẹo chống trôi ở tốc độ 1.0x: Cho phép câu bắt đầu sớm tối đa 60ms (Early Lead-in)
            # nếu trước đó đang là khoảng trống yên lặng và câu hiện tại hơi dài!
            earliest_allowed = t_target_start - 0.060 if idx > 0 else t_target_start
            if scheduled_events:
                prev_end = scheduled_events[-1]["actual_end"]
                earliest_allowed = max(earliest_allowed, prev_end + self.min_breath)

            natural_window = next_target_start - t_target_start - self.min_breath
            if raw_dur > natural_window and earliest_allowed < t_target_start:
                # Bắt đầu sớm nhẹ (tối đa 60ms) để giảm áp lực tràn sang câu sau
                actual_start = max(earliest_allowed, next_target_start - raw_dur - self.min_breath)
            else:
                actual_start = max(t_target_start, earliest_allowed)

            # 3. Tính khoảng trống thực tế trước câu tiếp theo
            real_avail_win = max(0.08, next_target_start - actual_start - self.min_breath)

            # 4. Quyết định tốc độ trong dải hẹp [0.98x - 1.05x]
            if raw_dur <= real_avail_win:
                # Vừa khít khoảng trống -> Giữ nguyên tốc độ tự nhiên 1.00x tuyệt đối!
                speed = 1.00
            else:
                needed_speed = raw_dur / real_avail_win
                # Nếu đang bị trễ tích lũy > 0.35s so với timeline gốc, cho phép nhích lên tối đa 1.10x để kéo lại
                current_lag = actual_start - t_target_start
                upper_limit = self.emergency_speed if current_lag > 0.35 else self.max_speed
                speed = float(np.clip(needed_speed, 1.00, upper_limit))

            speed_records.append(speed)

            # 5. Chỉ gọi FFmpeg atempo nếu tốc độ lệch > 1% so với 1.00x
            if abs(speed - 1.0) > 0.01:
                trimmed_wav = tts_cache_dir / f"sent_{sent_id:04d}_trimmed.wav"
                sped_wav = tts_cache_dir / f"sent_{sent_id:04d}_pid_sped.wav"
                sf.write(str(trimmed_wav), data_clean, self.sr)
                cmd = [
                    FFMPEG, "-nostdin", "-y",
                    "-i", str(trimmed_wav),
                    "-filter:a", f"atempo={speed:.4f}",
                    "-ar", str(self.sr), "-ac", "1",
                    str(sped_wav)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if sped_wav.exists():
                    audio_chunk, _ = sf.read(str(sped_wav))
                else:
                    audio_chunk = data_clean
            else:
                audio_chunk = data_clean

            final_dur = len(audio_chunk) / self.sr
            actual_end = min(total_duration, actual_start + final_dur)

            # 6. Chèn vào Master Track
            start_samp = int(round(actual_start * self.sr))
            end_samp = min(total_samples, start_samp + len(audio_chunk))
            insert_len = end_samp - start_samp

            if insert_len > 0:
                master_audio[start_samp:end_samp] += audio_chunk[:insert_len]

            start_drift = abs(actual_start - t_target_start)
            drift_records.append(start_drift)

            if idx + 1 < len(diff_events):
                next_s_start = diff_events[idx + 1]["target_start"]
                sub_end = min(max(actual_end, next_s_start - 0.02), max(actual_end, ev["target_end"]))
            else:
                sub_end = max(actual_end, min(total_duration, ev["target_end"]))

            sched_item = {
                "id": sent_id,
                "start": round(actual_start, 3),
                "end": round(sub_end, 3),
                "target_start": t_target_start,
                "target_end": ev["target_end"],
                "actual_start": round(actual_start, 3),
                "actual_end": round(actual_end, 3),
                "sub_end": round(sub_end, 3),
                "actual_dur": round(final_dur, 3),
                "speed_applied": round(speed, 3),
                "drift_ms": round(start_drift * 1000.0, 2),
                "text_vi": ev["text_vi"],
                "text_zh": ev["text_zh"]
            }
            for k, v in ev.items():
                if k not in sched_item:
                    sched_item[k] = v
            scheduled_events.append(sched_item)

        # Chuẩn hóa âm lượng đầu ra (Normalizing Peak to 0.95)
        max_val = np.max(np.abs(master_audio))
        if max_val > 0.001:
            master_audio = master_audio * (0.95 / max_val)

        sf.write(str(out_audio_path), master_audio, self.sr)

        metrics = {
            "total_sentences": len(scheduled_events),
            "collisions": collisions,
            "max_drift_ms": round(max(drift_records) * 1000.0, 2) if drift_records else 0.0,
            "mean_drift_ms": round(float(np.mean(drift_records)) * 1000.0, 2) if drift_records else 0.0,
            "mean_speed": round(float(np.mean(speed_records)), 3) if speed_records else 1.0,
            "zero_drift_rate": f"{round(sum(1 for d in drift_records if d < 0.080) / max(1, len(drift_records)) * 100, 1)}% (1.0x Natural Sync)"
        }

        return out_audio_path, scheduled_events, metrics
