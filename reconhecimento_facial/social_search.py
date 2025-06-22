import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

SOCIAL_MAPPER_REPO = "https://github.com/Greenwolf/social_mapper"
DEFAULT_CLONE_DIR = Path.home() / ".social_mapper"


def _clone_repo(dest: Path) -> Path:
    """Clone the social_mapper repository if not present."""
    if (dest / "social_mapper.py").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning social_mapper into %s", dest)
    subprocess.run(["git", "clone", SOCIAL_MAPPER_REPO, str(dest)], check=True)
    return dest


def ensure_repo(path: str | None = None) -> Path:
    path = Path(path or os.environ.get("SOCIAL_MAPPER_PATH", DEFAULT_CLONE_DIR))
    return _clone_repo(path)


def run_social_search(
    images: Sequence[str],
    name: str | None = None,
    sites: Iterable[str] = ("facebook",),
    fast: bool = True,
    repo_path: str | None = None,
) -> None:
    """Run social_mapper to search for a face on social networks."""
    repo = ensure_repo(repo_path)
    img_dir = repo / "input-images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        shutil.copy(img, img_dir)

    cmd = [
        sys.executable,
        str(repo / "social_mapper.py"),
        "-f",
        "imagefolder",
        "-i",
        str(img_dir),
        "-m",
        "fast" if fast else "accurate",
    ]
    for site in sites:
        cmd.append(f"--{site}")
    if name:
        cmd.extend(["-n", name])

    logger.info("Running social_mapper: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search social networks for a face")
    parser.add_argument("--image", action="append", required=True, help="Image to search")
    parser.add_argument("--name", help="Person name")
    parser.add_argument("--site", action="append", default=["facebook"], help="Site to search (facebook, twitter, instagram, etc.)")
    parser.add_argument("--accurate", action="store_true", help="Use accurate mode instead of fast")
    parser.add_argument("--repo", help="Path to social_mapper repository")
    args = parser.parse_args(argv)

    run_social_search(args.image, args.name, args.site, not args.accurate, args.repo)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
