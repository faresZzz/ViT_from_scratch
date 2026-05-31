"""Download a tiny set of real external images for qualitative run figures.

The downloaded files live under ``data/external_images/``, which is ignored by Git.
They are meant for notebook inspection only, not as a training dataset.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external_images"

EXTERNAL_IMAGES: tuple[tuple[str, str], ...] = (
    (
        "cat_domestic.jpg",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Domestic_Cat.jpg",
    ),
    (
        "airplane_commercial.jpg",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Commercial_jet_airplane.jpg",
    ),
    (
        "horse_camargue.jpg",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Camargue_horse.jpg",
    ),
    (
        "truck_wide_load.jpg",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Wide_Load_Truck.JPG",
    ),
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in EXTERNAL_IMAGES:
        output_path = OUTPUT_DIR / filename
        if output_path.exists():
            print(f"exists: {output_path}")
            continue
        print(f"download: {filename}")
        request = Request(url, headers={"User-Agent": "vit-from-scratch/0.1"})
        with urlopen(request, timeout=30) as response:
            output_path.write_bytes(response.read())
    print(f"External images ready in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
