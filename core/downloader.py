"""
Đồng bộ hóa: downloader.py chuyển hướng sang core/bilibili_downloader.py
để đảm bảo tính tương thích tuyệt đối cho toàn bộ hệ thống.
"""
from core.bilibili_downloader import (
    clean_bilibili_srt,
    extract_bvid_and_part,
    download_bilibili_bundle
)

__all__ = ["clean_bilibili_srt", "extract_bvid_and_part", "download_bilibili_bundle"]
