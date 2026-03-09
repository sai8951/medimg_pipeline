from pathlib import Path


def ensure_suffix(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if path.suffix != suffix:
        return path.with_suffix(suffix)
    return path
