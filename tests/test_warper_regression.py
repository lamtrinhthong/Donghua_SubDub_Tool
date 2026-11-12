"""
Test hồi quy cho Micro-PID Time Warper + Bilibili Downloader config.
Chạy:  python tests/test_warper_regression.py
"""
import sys
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.micro_pid_time_warper import (  # noqa: E402
    MicroPIDTimeWarper,
    trim_and_compact_speech,
    normalize_master,
    _atempo_chain,
)
from core.common import ffmpeg_location_for_ytdlp, ensure_ffmpeg_bin  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}{(' -> ' + detail) if detail else ''}")


def make_speech_wav(path, dur=1.0, sr=48000, tone=220.0, gap_at=None, flat=False):
    """Tạo WAV giả lập giọng nói: tone có envelope + khoảng lặng ở giữa."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    if flat:
        env = np.ones(n)
    else:
        env = np.sin(np.pi * np.clip(t / max(dur, 1e-6), 0, 1)) ** 0.5
    sig = 0.5 * env * np.sin(2 * np.pi * tone * t)
    if gap_at is not None:
        g0 = int(gap_at * sr)
        g1 = min(n, g0 + int(0.30 * sr))
        sig[g0:g1] = 0.0
    sf.write(str(path), sig.astype(np.float32), sr)
    return path


def test_no_overlap():
    print("\n=== TEST 1: Hai cau KHONG BAO GIỜ chong lan ===")
    tmp = Path(tempfile.mkdtemp())
    sr = 48000
    events = [
        {"id": 1, "start": 0.0, "end": 2.0, "text_vi": "Cau mot"},
        {"id": 2, "start": 2.0, "end": 4.0, "text_vi": "Cau hai rat dai"},
        {"id": 3, "start": 4.0, "end": 6.0, "text_vi": "Cau ba"},
    ]
    for ev in events:
        # Audio dai hon nhieu so voi khung thoi gian -> phai tua nhanh hoac cat
        make_speech_wav(tmp / f"sent_{ev['id']:04d}.wav", dur=3.5, sr=sr, flat=True)

    w = MicroPIDTimeWarper(sample_rate=sr, min_speed=1.0, max_speed=1.05,
                           emergency_speed=1.20, hard_cap_speed=1.40,
                           min_breath_pause=0.025)
    diff = w.differentiate_timeline(events, total_duration=8.0)
    out, sched, metrics = w.process_and_synthesize(
        diff, tmp, 8.0, tmp / "master.wav")

    ok = True
    for i in range(len(sched) - 1):
        a, b = sched[i], sched[i + 1]
        gap = b["actual_start"] - a["actual_end"]
        if gap < -0.001:
            ok = False
            print(f"      OVERLAP: [{a['id']}] end={a['actual_end']} > [{b['id']}] start={b['actual_start']}")
    check("Khong co overlap giua cac cau", ok)
    check("collisions > 0 (phat hien day duoc xung dot)",
          metrics["collisions"] > 0, f"collisions={metrics['collisions']}")
    check("toc do khong vuot qua hard cap",
          all(s["speed_applied"] <= 1.40 + 1e-6 for s in sched),
          f"max={metrics['max_speed_used']}")
    check("master audio khong vuot qua tong thoi luong",
          len(sf.read(str(out))[0]) <= int(8.0 * sr))
    shutil.rmtree(tmp, ignore_errors=True)


def test_hard_trim_when_impossible():
    print("\n=== TEST 1b: Audio vuot qua ca hard cap -> phai cat cuong, KHONG chong lan ===")
    tmp = Path(tempfile.mkdtemp())
    sr = 48000
    events = [
        {"id": 1, "start": 0.0, "end": 1.0, "text_vi": "Rat dai"},
        {"id": 2, "start": 1.0, "end": 2.0, "text_vi": "Ke tiep"},
    ]
    for ev in events:
        make_speech_wav(tmp / f"sent_{ev['id']:04d}.wav", dur=8.0, sr=sr, flat=True)
    w = MicroPIDTimeWarper(sample_rate=sr)
    diff = w.differentiate_timeline(events, total_duration=4.0)
    out, sched, metrics = w.process_and_synthesize(diff, tmp, 4.0, tmp / "m.wav")
    check("trimmed_sentences > 0 (cat cuong khi tran khung)",
          metrics["trimmed_sentences"] > 0, f"trimmed={metrics['trimmed_sentences']}")
    ok = True
    for a, b in zip(sched, sched[1:]):
        if b["actual_start"] - a["actual_end"] < -0.001:
            ok = False
    check("van khong chong lan sau khi cat cuong", ok)
    shutil.rmtree(tmp, ignore_errors=True)


def test_speed_band():
    print("\n=== TEST 2: Toc do nam trong dai [1.00 - 1.40] ===")
    tmp = Path(tempfile.mkdtemp())
    sr = 48000
    events = [
        {"id": 1, "start": 0.0, "end": 3.0, "text_vi": "Vua khop"},
        {"id": 2, "start": 3.0, "end": 6.0, "text_vi": "Can gap thoi gian"},
    ]
    make_speech_wav(tmp / "sent_0001.wav", dur=2.5, sr=sr, flat=True)
    make_speech_wav(tmp / "sent_0002.wav", dur=2.5, sr=sr, flat=True)
    w = MicroPIDTimeWarper(sample_rate=sr)
    diff = w.differentiate_timeline(events, total_duration=8.0)
    out, sched, metrics = w.process_and_synthesize(diff, tmp, 8.0, tmp / "m.wav")
    for s in sched:
        check(f"speed cau {s['id']} trong dai",
              1.0 - 1e-6 <= s["speed_applied"] <= 1.40 + 1e-6,
              f"speed={s['speed_applied']}")
    check("mean_speed <= 1.40", metrics["mean_speed"] <= 1.40,
          f"mean={metrics['mean_speed']}")
    shutil.rmtree(tmp, ignore_errors=True)


def test_sub_end_matches_audio():
    print("\n=== TEST 3: sub_end khop voi audio that ===")
    tmp = Path(tempfile.mkdtemp())
    sr = 48000
    events = [{"id": 1, "start": 0.0, "end": 1.0, "text_vi": "Ngan"}]
    make_speech_wav(tmp / "sent_0001.wav", dur=2.5, sr=sr)
    w = MicroPIDTimeWarper(sample_rate=sr)
    diff = w.differentiate_timeline(events, total_duration=5.0)
    out, sched, metrics = w.process_and_synthesize(diff, tmp, 5.0, tmp / "m.wav")
    s = sched[0]
    check("sub_end == actual_end (audio that)",
          abs(s["sub_end"] - s["actual_end"]) < 0.01,
          f"sub_end={s['sub_end']} actual_end={s['actual_end']}")
    check("actual_dur khong vuot qua cua so toi next sentence",
          s["actual_dur"] <= (5.0 - s["actual_start"]) + 1e-6, f"dur={s['actual_dur']}")
    check("khong cham vao cau truoc (khong overlap)",
          s["actual_start"] >= s["target_start"] - 0.061, f"start={s['actual_start']}")
    shutil.rmtree(tmp, ignore_errors=True)


def test_trim_keeps_head_and_tail():
    print("\n=== TEST 4: trim giu nguyen dau/cuoi (khong cat mat am) ===")
    sr = 48000
    t = np.arange(int(1.0 * sr)) / sr
    sig = 0.5 * np.sin(2 * np.pi * 300 * t)
    out = trim_and_compact_speech(sig, sr, pad_ms=18, max_internal_pause_ms=55)
    check("do dai sau trim khong bi rut qua 20%",
          len(out) > len(sig) * 0.8, f"{len(sig)} -> {len(out)}")
    # So do bien do dau/cuoi
    check("am dau giu nguyen bien do",
          abs(float(np.max(np.abs(out[:int(0.05*sr)]))) -
              float(np.max(np.abs(sig[:int(0.05*sr)])))) < 0.05)
    shutil.rmtree(tempfile.mkdtemp(), ignore_errors=True)


def test_normalize():
    print("\n=== TEST 5: normalize theo RMS ===")
    # Master track co nhieu khoang lang -> RMS toan cuc thap, phai dung phan vi 95
    a = np.zeros(48000 * 3, dtype=np.float32)
    tone = (np.random.randn(48000).astype(np.float32) * 0.01)
    a[48000:96000] = tone
    out = normalize_master(a, target_rms_dbfs=-20.0, sr=48000)
    speech = out[48000:96000]
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(speech.astype(np.float64) ** 2))))
    check("RMS vung co tieng sau normalize ~ -20 dBFS", abs(rms_db + 20.0) < 1.5, f"{rms_db:.2f} dBFS")
    check("peak <= 0.95", float(np.max(np.abs(out))) <= 0.9501)


def test_atempo_chain():
    print("\n=== TEST 6: chuoi atempo hop le ===")
    check("atempo 1.05", _atempo_chain(1.05) == "atempo=1.0500")
    check("atempo 1.6 -> 1 tầng", _atempo_chain(1.6).count(",") == 0)
    check("atempo 3.0 -> 2 tầng", _atempo_chain(3.0).count(",") == 1,
          _atempo_chain(3.0))
    check("atempo 0.4 -> 2 tầng", _atempo_chain(0.4).count(",") == 1,
          _atempo_chain(0.4))


def test_ffmpeg_location():
    print("\n=== TEST 7: ffmpeg_location cho yt-dlp ===")
    ensure_ffmpeg_bin()
    loc = ffmpeg_location_for_ytdlp()
    check("tra ve duong dan khong None", loc is not None, str(loc))
    from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor
    pp = FFmpegPostProcessor()
    pp._paths = pp._determine_executables()
    # Simulate truyền ffmpeg_location
    import yt_dlp
    with yt_dlp.YoutubeDL({"ffmpeg_location": loc, "quiet": True}) as ydl:
        from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor as FP
        fp = FP(downloader=ydl)
        check("FFmpegPostProcessor.available == True", fp.available is True)
    from yt_dlp.downloader.external import FFmpegFD
    check("FFmpegFD.available == True (PATH check)", FFmpegFD.available() is True)


def test_bilibili_cookies():
    print("\n=== TEST 8: phat hien cookie Bilibili trong trinh duyet ===")
    from core.bilibili_downloader import detect_bilibili_browser_cookies
    res = detect_bilibili_browser_cookies()
    check("tim thay cookie dang nhap", res is not None, str(res))


if __name__ == "__main__":
    test_no_overlap()
    test_hard_trim_when_impossible()
    test_speed_band()
    test_sub_end_matches_audio()
    test_trim_keeps_head_and_tail()
    test_normalize()
    test_atempo_chain()
    test_ffmpeg_location()
    test_bilibili_cookies()

    print("\n" + "=" * 60)
    print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
    if FAIL:
        print("FAILED TESTS:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("ALL TESTS PASSED")
