import argparse
import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import face_recognition
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    face_recognition = None

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path(os.getenv("SOCIAL_DB_PATH", Path.home() / ".social_db"))
THRESHOLD = 0.6


def _get_encoding(image: str) -> np.ndarray | None:
    if face_recognition is None:
        return None
    img = face_recognition.load_image_file(image)
    enc = face_recognition.face_encodings(img)
    return enc[0] if enc else None


def run_social_search(
    images: Sequence[str],
    name: str | None = None,
    sites: Iterable[str] = ("facebook",),
    fast: bool = True,  # noqa: D417 - kept for backward compatibility
    db_path: str | None = None,
) -> Path:
    """Search for matching faces in local directories for each site.

    Returns the path to a temporary directory containing CSV files with matches.
    """
    base_dir = Path(db_path or DEFAULT_DB_DIR)
    out_dir = Path(tempfile.mkdtemp(prefix="social_search_"))

    for site in sites:
        site_dir = base_dir / site
        if not site_dir.exists():
            logger.warning("Directory %s not found", site_dir)
            continue
        results = []
        for img in images:
            enc = _get_encoding(img)
            if enc is None:
                continue
            for file in site_dir.glob("*"):
                if not file.is_file():
                    continue
                db_enc = _get_encoding(str(file))
                if db_enc is None:
                    continue
                dist = float(np.linalg.norm(db_enc - enc))
                if dist < THRESHOLD:
                    results.append({"image": str(file), "distance": dist})
        if results:
            csv_path = out_dir / f"{site}_{name or 'results'}.csv"
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["image", "distance"])
                writer.writeheader()
                writer.writerows(results)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Search local images for a face")
    parser.add_argument(
        "--image", action="append", required=True, help="Image to search"
    )
    parser.add_argument("--name", help="Person name")
    parser.add_argument(
        "--site",
        action="append",
        default=["facebook"],
        help="Site folder to search (facebook, twitter, etc.)",
    )
    parser.add_argument("--db", help="Path to local image database")
    args = parser.parse_args(argv)

    run_social_search(args.image, args.name, args.site, True, args.db)


if __name__ == "__main__":
    main()
