from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from column_profiler import parse_numeric
from field_scorer import TARGET_FIELDS


def default_output_path(input_path):
    path = Path(input_path).expanduser()
    safe_stem = _safe_path_stem(path.stem)
    output_dir = _new_operation_output_dir(safe_stem)
    return output_dir / f"{safe_stem}_standardized.csv"


def save_standardized_dataset(df, mapping, output_path):
    standardized = build_standardized_dataframe(df, mapping)
    path = _ensure_csv_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    standardized.to_csv(path, index=False)

    return path


def build_standardized_dataframe(df, mapping):
    standardized = pd.DataFrame()

    for target in TARGET_FIELDS:
        source_column = mapping.get(target)
        if source_column is None and target == "revenue":
            standardized[target] = _derive_revenue(df, mapping)
        elif source_column is not None:
            standardized[target] = df[source_column]
        else:
            standardized[target] = pd.NA

    return standardized[TARGET_FIELDS]


def _ensure_csv_path(output_path):
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    return path


def _new_operation_output_dir(stem):
    project_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = project_dir / f"islem_{stem}_{timestamp}"
    output_dir = base_dir
    suffix = 2

    while output_dir.exists():
        output_dir = project_dir / f"{base_dir.name}_{suffix}"
        suffix += 1

    return output_dir


def _safe_path_stem(stem):
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe_stem or "veri"


def _derive_revenue(df, mapping):
    quantity_column = mapping.get("quantity")
    unit_price_column = mapping.get("unit_price")

    if quantity_column is None or unit_price_column is None:
        return pd.Series(pd.NA, index=df.index)

    quantity = df[quantity_column].map(parse_numeric)
    unit_price = df[unit_price_column].map(parse_numeric)

    return pd.to_numeric(quantity, errors="coerce") * pd.to_numeric(
        unit_price,
        errors="coerce",
    )
