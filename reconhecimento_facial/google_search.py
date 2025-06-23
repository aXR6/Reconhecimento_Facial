import argparse
import logging
import webbrowser
from typing import Sequence

import requests

logger = logging.getLogger(__name__)


def run_google_search(images: Sequence[str]) -> list[str]:
    """Search each image on Google Images and open the results page.

    A small text interface is printed to the terminal so the user knows the
    face is being used as the search target on Google and Google Images.
    """
    urls = []
    for img in images:
        try:
            msg = (
                f"\n{'=' * 60}\n"
                f"O rosto em '{img}' esta sendo buscado no Google e no Google Imagens"
                f"\n{'=' * 60}"
            )
            print(msg)
            with open(img, "rb") as fh:
                files = {"encoded_image": fh}
                resp = requests.post(
                    "https://www.google.com/searchbyimage/upload",
                    files=files,
                    allow_redirects=False,
                )
            url = resp.headers.get("Location")
            if resp.status_code == 302 and url:
                urls.append(url)
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            else:
                logger.warning("Search failed for %s", img)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("google search error: %s", exc)
    return urls


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Search images on Google")
    parser.add_argument(
        "--image", action="append", required=True, help="Image to search"
    )
    args = parser.parse_args(argv)
    run_google_search(args.image)


if __name__ == "__main__":
    main()
