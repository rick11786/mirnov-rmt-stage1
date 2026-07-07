"""Sync viewable outputs to the Windows Documents mirror."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ROOT = Path("/mnt/c/Users/28105/Documents/mirnov_rmt_stage1")


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def copy_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.name == ".gitkeep":
            continue
        dst = dst_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, copy_function=shutil.copyfile)
        else:
            shutil.copyfile(src, dst)


def main() -> None:
    try:
        WINDOWS_ROOT.mkdir(parents=True, exist_ok=True)
        copy_file(ROOT / "report.html", WINDOWS_ROOT / "report.html")
        copy_tree_contents(ROOT / "figures", WINDOWS_ROOT / "figures")
        copy_tree_contents(ROOT / "results", WINDOWS_ROOT / "results")
    except OSError as exc:
        raise SystemExit(
            f"Could not sync to {WINDOWS_ROOT}: {exc}\n"
            "The current WSL mount exposes /mnt/c as read-only to this process. "
            "Open the report through \\\\wsl$ or rerun after enabling write access."
        ) from exc
    print(f"Synced outputs to {WINDOWS_ROOT}")


if __name__ == "__main__":
    main()
