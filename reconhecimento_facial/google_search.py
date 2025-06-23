import argparse
import logging
import webbrowser
from typing import Sequence

import requests
import time

logger = logging.getLogger(__name__)


def run_google_search(images: Sequence[str], *, show_status: bool = False) -> list[str]:
    """Search each image on Google Images and open the results page.

    A small text interface is printed to the terminal so the user knows the
    face is being used as the search target on Google and Google Images. When
    ``show_status`` is ``True`` a small Tkinter window is displayed indicating
    the progress of the search.
    """

    window = None
    status_var = None
    if show_status:
        try:  # pragma: no cover - best effort
            import tkinter as tk

            window = tk.Tk()
            window.title("Busca no Google")
            status_var = tk.StringVar()
            tk.Label(window, textvariable=status_var, padx=20, pady=20).pack()
            window.update_idletasks()
            window.update()
        except Exception as exc:  # pragma: no cover - GUI might not be available
            logger.error("unable to create status window: %s", exc)
            window = None
            status_var = None

    urls = []
    for img in images:
        try:
            msg = (
                f"\n{'=' * 60}\n"
                f"O rosto em '{img}' esta sendo buscado no Google e no Google Imagens"
                f"\n{'=' * 60}"
            )
            print(msg)
            if status_var is not None:
                status_var.set(f"Buscando {img} ...")
                window.update_idletasks()
                window.update()
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
                opened = False
                try:
                    opened = webbrowser.open(url)
                except Exception as exc:  # pragma: no cover - browser may not exist
                    logger.error("failed to open webbrowser: %s", exc)
                if not opened:
                    print(f"\nAbra este link manualmente:\n{url}")
            else:
                logger.warning("Search failed for %s", img)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("google search error: %s", exc)

    if status_var is not None and window is not None:
        status_var.set("Busca finalizada")
        window.update_idletasks()
        window.update()
        time.sleep(1)
        window.destroy()
    return urls


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Search images on Google")
    parser.add_argument(
        "--image", action="append", required=True, help="Image to search"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show search progress in a window"
    )
    args = parser.parse_args(argv)
    run_google_search(args.image, show_status=args.status)


if __name__ == "__main__":
    main()
