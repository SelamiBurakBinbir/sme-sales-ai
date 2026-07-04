from pathlib import Path

import pandas as pd


def read_dataset(file_path):
    """Read a CSV or XLSX file and return a pandas DataFrame."""
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Sadece CSV, XLSX veya XLS dosyalari destekleniyor.")


def _read_csv(path):
    encodings = ["utf-8", "utf-8-sig", "cp1254", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error
