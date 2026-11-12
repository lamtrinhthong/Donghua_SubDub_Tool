import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

QUEUE_FILE = Path("D:/FastLab_AI_Studio/series_queue.json")


def load_queue_status() -> Dict[str, Any]:
    if QUEUE_FILE.exists():
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "studio": "FastLab AI Studio",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_active_completed_volumes": 0,
        "series": {}
    }


def update_volume_completed(
    series_name: str,
    completed_vol: int,
    next_vol: Optional[int] = None,
    total_vols: Optional[int] = None,
    video_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Cập nhật trạng thái sau khi hoàn thành 1 Volume theo đúng Quy Chuẩn Rule 7:
    - Thêm số volume vừa hoàn thành vào completed_volumes (sort tăng dần).
    - Cập nhật in_progress_volume sang volume tiếp theo.
    - Loại bỏ volume đã xong khỏi remaining_volumes.
    - Cập nhật total_active_completed_volumes và last_updated.
    - Chuyển status sang COMPLETED_100% khi toàn bộ volume hoàn tất.
    """
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = load_queue_status()

    if "series" not in data:
        data["series"] = {}

    series_data = data["series"].get(series_name, {
        "status": "IN_PROGRESS",
        "completed_volumes": [],
        "in_progress_volume": completed_vol,
        "remaining_volumes": [],
        "total_volumes": total_vols or completed_vol,
        "volumes_info": {}
    })

    completed_list = set(series_data.get("completed_volumes", []))
    completed_list.add(completed_vol)
    sorted_completed = sorted(list(completed_list))
    series_data["completed_volumes"] = sorted_completed

    if total_vols:
        series_data["total_volumes"] = max(series_data.get("total_volumes", 0), total_vols)

    # Cập nhật remaining
    all_vols = set(range(1, series_data["total_volumes"] + 1))
    remaining = sorted(list(all_vols - completed_list))
    if next_vol and next_vol in remaining:
        remaining.remove(next_vol)

    series_data["remaining_volumes"] = remaining
    series_data["in_progress_volume"] = next_vol

    if video_path:
        if "volumes_info" not in series_data:
            series_data["volumes_info"] = {}
        series_data["volumes_info"][f"vol_{completed_vol:02d}"] = {
            "video_path": str(video_path),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }

    if not remaining and next_vol is None:
        series_data["status"] = "COMPLETED_100%"
    else:
        series_data["status"] = "IN_PROGRESS"

    data["series"][series_name] = series_data

    # Tính tổng volume hoàn thành toàn Studio
    total_comp = sum(len(s.get("completed_volumes", [])) for s in data["series"].values())
    data["total_active_completed_volumes"] = total_comp
    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[*] [Queue Status] Đã cập nhật {series_name} Vol {completed_vol} -> series_queue.json")
    return data
