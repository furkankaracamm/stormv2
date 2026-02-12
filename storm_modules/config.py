from __future__ import annotations

import os
from pathlib import Path


def get_work_dir() -> Path:
    env_dir = os.environ.get("STORM_WORK_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.cwd()


def get_academic_brain_db_path() -> Path:
    env_path = os.environ.get("STORM_ACADEMIC_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return get_work_dir() / "academic_brain.db"


def get_insights_db_path() -> Path:
    env_path = os.environ.get("STORM_INSIGHTS_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return get_work_dir() / "insights.db"
