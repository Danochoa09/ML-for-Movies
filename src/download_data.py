"""Descarga el dataset 'IMDb 100000 Movies/TV Shows' desde Kaggle (kagglehub)
y copia los CSV a data/raw/ para que el proyecto sea reproducible.

Uso:
    python src/download_data.py
"""
from pathlib import Path
import shutil

import kagglehub

DATASET = "kurtnakasato/imdb-100000-moviestvshows"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Descargando '{DATASET}' ...")
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Descargado en cache: {cache_path}")

    csvs = sorted(cache_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No se encontraron CSV en {cache_path}")

    for csv in csvs:
        dst = RAW_DIR / csv.name
        shutil.copy2(csv, dst)
        print(f"  -> {dst.relative_to(RAW_DIR.parents[1])} ({dst.stat().st_size/1e6:.1f} MB)")

    print(f"\nListo. {len(csvs)} archivos en {RAW_DIR}")


if __name__ == "__main__":
    main()
