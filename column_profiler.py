import re
import unicodedata
import warnings
from collections import Counter

import pandas as pd


CURRENCY_MARKERS = ("₺", "tl", "try", "$", "usd", "€", "eur")


def normalize_column_name(name):
    text = str(name).lower().strip()
    text = text.replace("i̇", "i")
    text = text.translate(str.maketrans({
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\s\-_]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def profile_dataframe(df, sample_size=10):
    profiles = {}
    for column in df.columns:
        sample = df[column].head(sample_size).dropna().tolist()
        profiles[column] = profile_column(column, sample)
    return profiles


def profile_column(column_name, sample_values):
    value_types = [_classify_value(value) for value in sample_values]
    dominant_type = _dominant_type(value_types, sample_values)

    return {
        "column": column_name,
        "normalized_name": normalize_column_name(column_name),
        "sample_values": sample_values,
        "value_types": value_types,
        "dominant_type": dominant_type,
        "has_currency": has_currency_marker(column_name, sample_values),
    }


def has_currency_marker(column_name, sample_values):
    joined = " ".join([str(column_name), *[str(value) for value in sample_values]]).lower()
    normalized_joined = normalize_column_name(joined)

    return any(marker in joined for marker in CURRENCY_MARKERS) or any(
        marker in normalized_joined for marker in ("tl", "try", "usd", "eur")
    )


def parse_numeric(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    text = text.lower()
    for marker in CURRENCY_MARKERS:
        text = text.replace(marker, "")

    if re.search(r"[a-zA-Z]", text):
        return None

    text = re.sub(r"[^0-9,\.\-]", "", text)

    if not re.search(r"\d", text):
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def _classify_value(value):
    if pd.isna(value):
        return "empty"

    if _is_date_like(value):
        return "date"

    if parse_numeric(value) is not None:
        return "numeric"

    return "string"


def _is_date_like(value):
    if pd.isna(value):
        return False

    if isinstance(value, pd.Timestamp):
        return True

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False

    text = str(value).strip()
    if not text:
        return False

    if parse_numeric(text) is not None:
        return False

    if not re.search(r"\d", text):
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return not pd.isna(parsed)


def _dominant_type(value_types, sample_values):
    filtered_types = [value_type for value_type in value_types if value_type != "empty"]
    if not filtered_types:
        return "short_string"

    counter = Counter(filtered_types)
    dominant, _ = counter.most_common(1)[0]

    if dominant == "string":
        string_values = [
            str(value)
            for value, value_type in zip(sample_values, value_types)
            if value_type == "string" and not pd.isna(value)
        ]
        average_length = (
            sum(len(value) for value in string_values) / len(string_values)
            if string_values
            else 0
        )
        return "long_string" if average_length > 10 else "short_string"

    return dominant
