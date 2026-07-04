import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import dateparser
import ftfy
import pandas as pd
import pandera.pandas as pa
from price_parser import Price

from field_scorer import TARGET_FIELDS


# CLEANING_LOG_FILE = "data_cleaning_debug.log"
DATE_ORDER_CONFIDENCE_THRESHOLD = 0.99
REVENUE_ABSOLUTE_TOLERANCE = 0.05
PLACEHOLDER_VALUES = {
    "",
    "-",
    "--",
    "---",
    "?",
    "??",
    "unknown",
    "null",
    "none",
    "nan",
    "na",
    "n/a",
    "not available",
    "not applicable",
    "missing",
    "undefined",
    "empty",
    "blank",
    "yok",
    "bilinmiyor",
    "bilinmeyen",
    "bos",
    "boş",
}


def _normalize_discount_value(value):
    if pd.isna(value):
        return pd.NA
    if value > 1 and value <= 100:
        return value / 100
    return value


@dataclass
class CleaningResult:
    cleaned_path: Path | None
    report_path: Path | None
    dataframe: pd.DataFrame
    report: dict


def default_cleaned_output_path(standardized_path):
    path = Path(standardized_path).expanduser()
    stem = path.stem
    if stem.endswith("_standardized"):
        stem = stem.removesuffix("_standardized")
    return path.with_name(f"{stem}_cleaned.csv")


def clean_standardized_file(
    standardized_path,
    cleaned_output_path=None,
    # log_path=CLEANING_LOG_FILE,
):
    logger = logging.getLogger(__name__)
    input_path = Path(standardized_path).expanduser()
    output_path = (
        Path(cleaned_output_path).expanduser()
        if cleaned_output_path
        else default_cleaned_output_path(input_path)
    ).with_suffix(".csv")

    logger.info("Reading standardized dataset: %s", input_path)
    raw_df = pd.read_csv(input_path, dtype=object)
    cleaner = StandardizedDataCleaner(logger=logger)
    cleaned_df, report = cleaner.clean(raw_df)

    schema = build_standardized_schema()
    try:
        schema.validate(cleaned_df, lazy=True)
        report["schema_validation"] = {"success": True, "errors": []}
        logger.info("Pandera schema validation succeeded")
    except pa.errors.SchemaErrors as exc:
        errors = exc.failure_cases.to_dict(orient="records")
        report["schema_validation"] = {"success": False, "errors": errors}
        logger.warning("Pandera schema validation failed: %s", errors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    report_path = output_path.with_name(f"{output_path.stem}_cleaning_report.json")
    report["input_path"] = str(input_path)
    report["output_path"] = str(output_path)
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Cleaned dataset saved to: %s", output_path)
    logger.info("Cleaning report saved to: %s", report_path)

    return CleaningResult(
        cleaned_path=output_path,
        report_path=report_path,
        dataframe=cleaned_df,
        report=report,
    )


def clean_standardized_dataframe(df, discount_series=None):
    cleaner = StandardizedDataCleaner(logger=logging.getLogger(__name__))
    cleaned_df, report = cleaner.clean(df.copy(), discount_series=discount_series)

    schema = build_standardized_schema()
    try:
        schema.validate(cleaned_df, lazy=True)
        report["schema_validation"] = {"success": True, "errors": []}
    except pa.errors.SchemaErrors as exc:
        report["schema_validation"] = {
            "success": False,
            "errors": exc.failure_cases.to_dict(orient="records"),
        }

    return CleaningResult(
        cleaned_path=None,
        report_path=None,
        dataframe=cleaned_df,
        report=_json_safe(report),
    )


class StandardizedDataCleaner:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def clean(self, df, discount_series=None):
        self._validate_columns(df)

        cleaned = pd.DataFrame(index=df.index)
        report = {
            "total_rows": len(df),
            "columns": {},
            "quality_checks": {},
        }

        cleaned["date"], report["columns"]["date"] = self._clean_date(df["date"])
        cleaned["product_name"], report["columns"]["product_name"] = (
            self._clean_product_name(df["product_name"])
        )
        cleaned["quantity"], report["columns"]["quantity"] = self._clean_quantity(
            df["quantity"]
        )
        cleaned["unit_price"], report["columns"]["unit_price"] = self._clean_money(
            df["unit_price"],
            "unit_price",
        )
        cleaned["revenue"], report["columns"]["revenue"] = self._clean_revenue(
            df["revenue"],
            cleaned["quantity"],
            cleaned["unit_price"],
        )
        cleaned_discount = (
            self._clean_discount(discount_series)
            if discount_series is not None
            else None
        )
        report["quality_checks"]["revenue_mismatch"] = self._check_revenue_mismatch(
            cleaned["quantity"],
            cleaned["unit_price"],
            cleaned["revenue"],
            cleaned_discount,
        )

        for column in TARGET_FIELDS:
            column_report = report["columns"][column]
            self.logger.info(
                "%s converted: %s success, %s failed, dtype=%s",
                column,
                column_report["converted_count"],
                column_report["failed_count"],
                column_report["target_dtype"],
            )

        self._log_quality_checks(report)
        return cleaned[TARGET_FIELDS], report

    def _validate_columns(self, df):
        missing_columns = [column for column in TARGET_FIELDS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"The standardized dataset is missing columns: {missing_columns}")

    def _clean_date(self, series):
        text_values = series.map(_clean_text_value)
        date_detection = _detect_date_order(text_values)
        cleaned = _parse_dates_fast(text_values, date_detection)

        report = _build_column_report(series, cleaned, "date")
        report["date_format_detection"] = date_detection
        if date_detection["status"] == "inferred":
            report["business_error_count"] = date_detection["invalid_row_count"]
            report["business_error_examples"] = date_detection["invalid_row_examples"]
        else:
            report["business_error_count"] = report["failed_count"]
            report["business_error_examples"] = report["failed_examples"]
        return cleaned, report

    def _clean_product_name(self, series):
        cleaned = series.map(_clean_product_name_value).astype("string")
        report = _build_column_report(series, cleaned, "product_name")
        report["placeholder_count"] = int((~series.isna() & cleaned.isna()).sum())
        return cleaned, report

    def _clean_quantity(self, series):
        cleaned = self._clean_float(series)
        invalid_mask = cleaned.notna() & (cleaned <= 0)
        invalid_examples = _examples_from_mask(series, invalid_mask)
        if invalid_mask.any():
            cleaned.loc[invalid_mask] = pd.NA

        cleaned = cleaned.astype("float64")
        report = _build_column_report(series, cleaned, "quantity")
        report["non_positive_count"] = int(invalid_mask.sum())
        report["business_error_count"] = int(invalid_mask.sum())
        report["business_error_examples"] = invalid_examples
        return cleaned, report

    def _clean_money(self, series, column_name):
        cleaned = self._clean_float(series).round(2).astype("float64")
        report = _build_column_report(series, cleaned, column_name)
        report["rounded_to_decimal_places"] = 2
        return cleaned, report

    def _clean_revenue(self, series, quantity, unit_price):
        original_cleaned = self._clean_float(series)
        missing_before_fill = original_cleaned.isna()
        derived_revenue = (quantity * unit_price).round(2)
        cleaned = original_cleaned.fillna(derived_revenue).round(2).astype("float64")
        fill_mask = missing_before_fill & derived_revenue.notna()

        report = _build_column_report(series, cleaned, "revenue")
        report["rounded_to_decimal_places"] = 2
        report["filled_from_quantity_unit_price_count"] = int(fill_mask.sum())
        report["filled_from_quantity_unit_price_examples"] = [
            {
                "row_index": int(index),
                "quantity": _safe_scalar(quantity.loc[index]),
                "unit_price": _safe_scalar(unit_price.loc[index]),
                "derived_revenue": _safe_scalar(derived_revenue.loc[index]),
            }
            for index in series[fill_mask].head(10).index
        ]
        report["missing_after_fill_count"] = int(cleaned.isna().sum())
        return cleaned, report

    def _clean_discount(self, series):
        cleaned = self._clean_float(series)
        return cleaned.map(_normalize_discount_value).astype("float64")

    def _clean_float(self, series):
        text_values = series.map(_clean_text_value)
        text_values = text_values.mask(text_values.map(_is_placeholder_text))
        normalized_values = text_values.map(_normalize_numeric_text)
        cleaned = pd.to_numeric(normalized_values, errors="coerce")

        failed_mask = cleaned.isna() & text_values.notna()
        if failed_mask.any():
            fallback = text_values[failed_mask].map(_parse_price_like_number)
            cleaned.loc[failed_mask] = pd.to_numeric(fallback, errors="coerce")

        return cleaned.astype("float64")

    def _check_revenue_mismatch(self, quantity, unit_price, revenue, discount=None):
        reports = [
            self._build_revenue_mismatch_report(
                quantity,
                unit_price,
                revenue,
                discount=None,
                formula="quantity * unit_price",
                formula_label="quantity × unit_price",
            )
        ]
        if discount is not None:
            reports.append(
                self._build_revenue_mismatch_report(
                    quantity,
                    unit_price,
                    revenue,
                    discount=discount,
                    formula="quantity * unit_price * (1 - discount)",
                    formula_label="quantity × unit_price × (1 - discount)",
                )
            )

        best_report = max(
            reports,
            key=lambda report: (
                report["match_ratio"],
                report["match_count"],
                report["checked_rows"],
            ),
        )
        best_report["formula_results"] = [
            {
                "formula": report["formula"],
                "formula_label": report["formula_label"],
                "checked_rows": report["checked_rows"],
                "match_count": report["match_count"],
                "mismatch_count": report["mismatch_count"],
                "match_ratio": report["match_ratio"],
            }
            for report in reports
        ]
        return best_report

    def _build_revenue_mismatch_report(
        self,
        quantity,
        unit_price,
        revenue,
        discount,
        formula,
        formula_label,
    ):
        expected = quantity * unit_price
        checked_mask = quantity.notna() & unit_price.notna() & revenue.notna()
        if discount is not None:
            expected = expected * (1 - discount)
            checked_mask = checked_mask & discount.notna()

        expected = expected.round(2)
        absolute_error = (expected - revenue).abs()
        mismatch_mask = checked_mask & (absolute_error > REVENUE_ABSOLUTE_TOLERANCE)
        checked_rows = int(checked_mask.sum())
        mismatch_count = int(mismatch_mask.sum())
        match_count = checked_rows - mismatch_count
        match_ratio = match_count / checked_rows if checked_rows else 0

        return {
            "checked_rows": checked_rows,
            "match_count": match_count,
            "mismatch_count": mismatch_count,
            "match_ratio": match_ratio,
            "formula": formula,
            "formula_label": formula_label,
            "absolute_tolerance": REVENUE_ABSOLUTE_TOLERANCE,
            "mismatch_examples": [
                {
                    "row_index": int(index),
                    "quantity": _safe_scalar(quantity.loc[index]),
                    "unit_price": _safe_scalar(unit_price.loc[index]),
                    "discount": _safe_scalar(discount.loc[index]) if discount is not None else None,
                    "expected_revenue": _safe_scalar(expected.loc[index]),
                    "actual_revenue": _safe_scalar(revenue.loc[index]),
                    "absolute_error": _safe_scalar(absolute_error.loc[index]),
                }
                for index in revenue[mismatch_mask].head(20).index
            ],
        }

    def _log_quality_checks(self, report):
        date_detection = report["columns"]["date"]["date_format_detection"]
        self.logger.info(
            "date format detection: status=%s, order=%s, confidence=%s",
            date_detection["status"],
            date_detection["date_order"],
            date_detection["confidence"],
        )

        quantity_report = report["columns"]["quantity"]
        self.logger.info(
            "quantity non-positive errors: %s",
            quantity_report["non_positive_count"],
        )

        revenue_report = report["columns"]["revenue"]
        self.logger.info(
            "revenue filled from quantity * unit_price: %s",
            revenue_report["filled_from_quantity_unit_price_count"],
        )

        mismatch_report = report["quality_checks"]["revenue_mismatch"]
        self.logger.info(
            "revenue mismatch check: %s checked, %s mismatched",
            mismatch_report["checked_rows"],
            mismatch_report["mismatch_count"],
        )


def build_standardized_schema():
    return pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, nullable=True),
            "product_name": pa.Column(pa.String, nullable=True),
            "quantity": pa.Column(pa.Float, nullable=True),
            "unit_price": pa.Column(pa.Float, nullable=True),
            "revenue": pa.Column(pa.Float, nullable=True),
        },
        strict=True,
        coerce=False,
    )


def _detect_date_order(text_values):
    extracted = _extract_date_token_frame(text_values)
    has_numeric_date = extracted["has_numeric_date"]
    left = extracted["left"]
    right = extracted["right"]

    left_gt_12_mask = has_numeric_date & left.gt(12)
    right_gt_12_mask = has_numeric_date & right.gt(12)
    both_gt_12_mask = left_gt_12_mask & right_gt_12_mask
    left_only_mask = left_gt_12_mask & ~right_gt_12_mask
    right_only_mask = right_gt_12_mask & ~left_gt_12_mask
    ambiguous_mask = has_numeric_date & ~left_gt_12_mask & ~right_gt_12_mask

    left_gt_12_rows = text_values.index[left_only_mask].tolist()
    right_gt_12_rows = text_values.index[right_only_mask].tolist()
    both_gt_12_count = int(both_gt_12_mask.sum())
    ambiguous_count = int(ambiguous_mask.sum())
    scanned_count = int(has_numeric_date.sum())
    both_gt_12_examples = [
        _date_error_example(index, text_values.loc[index], "both_sides_over_12")
        for index in text_values.index[both_gt_12_mask][:20]
    ]

    evidence_count = len(left_gt_12_rows) + len(right_gt_12_rows)
    confidence = 0
    date_order = None
    status = "undecidable_no_evidence"
    conflict_rows = []

    if evidence_count:
        if len(left_gt_12_rows) >= len(right_gt_12_rows):
            dominant_side = "left"
            dominant_count = len(left_gt_12_rows)
            conflict_rows = right_gt_12_rows
            date_order = "day_month"
        else:
            dominant_side = "right"
            dominant_count = len(right_gt_12_rows)
            conflict_rows = left_gt_12_rows
            date_order = "month_day"

        confidence = dominant_count / evidence_count
        if confidence >= DATE_ORDER_CONFIDENCE_THRESHOLD:
            status = "inferred"
        else:
            status = "undecidable_mixed_evidence"
            date_order = None
    else:
        dominant_side = None

    conflict_reason = (
        "conflicts_with_inferred_date_order"
        if status == "inferred"
        else "date_order_undecidable_mixed_evidence"
    )
    conflict_examples = [
        _date_error_example(index, text_values.loc[index], conflict_reason)
        for index in conflict_rows[:20]
    ]

    invalid_examples = both_gt_12_examples + conflict_examples
    invalid_row_count = both_gt_12_count + (len(conflict_rows) if status == "inferred" else 0)
    return {
        "status": status,
        "date_order": date_order,
        "dominant_side_over_12": dominant_side,
        "confidence": round(confidence, 6),
        "confidence_threshold": DATE_ORDER_CONFIDENCE_THRESHOLD,
        "scanned_numeric_date_rows": int(scanned_count),
        "evidence_rows": int(evidence_count),
        "left_side_over_12_count": int(len(left_gt_12_rows)),
        "right_side_over_12_count": int(len(right_gt_12_rows)),
        "ambiguous_numeric_date_count": int(ambiguous_count),
        "both_sides_over_12_count": int(both_gt_12_count),
        "invalid_row_count": int(invalid_row_count),
        "invalid_row_examples": invalid_examples[:20],
    }


def _parse_date_with_detected_order(value, date_detection):
    if _is_placeholder_text(value):
        return pd.NaT

    parts = _extract_numeric_date_parts(value)
    if parts:
        date_order = date_detection["date_order"]
        if date_order is None or _date_parts_conflict_with_order(parts, date_order):
            return pd.NaT

        year = parts["year"]
        if year is None:
            return _parse_text_date(value, date_order)

        if date_order == "day_month":
            day = parts["left"]
            month = parts["right"]
        else:
            month = parts["left"]
            day = parts["right"]

        try:
            return pd.Timestamp(year=int(year), month=int(month), day=int(day))
        except ValueError:
            return pd.NaT

    return _parse_text_date(value, date_detection["date_order"])


def _parse_dates_fast(text_values, date_detection):
    cleaned = pd.Series(pd.NaT, index=text_values.index, dtype="datetime64[ns]")
    date_order = date_detection["date_order"]
    numeric_mask = pd.Series(False, index=text_values.index)

    if date_order is not None:
        numeric_dates, numeric_mask = _parse_numeric_dates_fast(text_values, date_order)
        cleaned.loc[numeric_mask] = numeric_dates.loc[numeric_mask]

    fallback_mask = cleaned.isna() & text_values.notna() & ~numeric_mask & ~_placeholder_mask(text_values)
    if fallback_mask.any():
        fallback = text_values[fallback_mask].map(
            lambda value: _parse_text_date(value, date_order)
        )
        cleaned.loc[fallback_mask] = pd.to_datetime(fallback, errors="coerce")

    return cleaned.astype("datetime64[ns]")


def _parse_numeric_dates_fast(text_values, date_order):
    extracted = _extract_date_token_frame(text_values)
    has_numeric_date = extracted["has_numeric_date"]
    has_year = has_numeric_date & extracted["third_text"].notna()

    parsed = pd.Series(pd.NaT, index=text_values.index, dtype="datetime64[ns]")
    if not has_year.any():
        return parsed, has_numeric_date

    first = extracted.loc[has_year, "first"]
    third = extracted.loc[has_year, "third"]
    first_text = extracted.loc[has_year, "first_text"]
    third_text = extracted.loc[has_year, "third_text"]
    year_left = first_text.str.len().eq(4) | first.gt(31)

    year = third.where(~year_left, first).map(_normalize_year)
    left = extracted.loc[has_year, "left"]
    right = extracted.loc[has_year, "right"]

    if date_order == "day_month":
        day = left
        month = right
    else:
        month = left
        day = right

    valid = (
        year.notna()
        & month.between(1, 12)
        & day.between(1, 31)
        & ~_date_parts_conflict_series(left, right, date_order)
        & (third_text.str.len().le(2) | third_text.str.len().eq(4) | third.gt(31) | year_left)
    )

    frame = pd.DataFrame({
        "year": year[valid].astype("int64"),
        "month": month[valid].astype("int64"),
        "day": day[valid].astype("int64"),
    })
    parsed.loc[frame.index] = pd.to_datetime(frame, errors="coerce")
    return parsed, has_numeric_date


def _extract_date_token_frame(text_values):
    as_text = text_values.astype("string")
    extracted = as_text.str.extract(r"^\D*(\d+)\D+(\d+)(?:\D+(\d+))?", expand=True)
    extracted.columns = ["first_text", "second_text", "third_text"]

    placeholder_mask = _placeholder_mask(text_values)
    has_numeric_date = (
        ~placeholder_mask
        & extracted["first_text"].notna()
        & extracted["second_text"].notna()
    )

    first = pd.to_numeric(extracted["first_text"], errors="coerce")
    second = pd.to_numeric(extracted["second_text"], errors="coerce")
    third = pd.to_numeric(extracted["third_text"], errors="coerce")

    has_third = extracted["third_text"].notna()
    year_left = has_third & (extracted["first_text"].str.len().eq(4) | first.gt(31))
    left = first.where(~year_left, second)
    right = second.where(~year_left, third)

    return pd.DataFrame({
        "first_text": extracted["first_text"],
        "second_text": extracted["second_text"],
        "third_text": extracted["third_text"],
        "first": first,
        "second": second,
        "third": third,
        "left": left,
        "right": right,
        "has_numeric_date": has_numeric_date,
    }, index=text_values.index)


def _date_parts_conflict_series(left, right, date_order):
    left_gt_12 = left > 12
    right_gt_12 = right > 12
    if date_order == "day_month":
        return right_gt_12 | (left_gt_12 & right_gt_12)
    return left_gt_12 | (left_gt_12 & right_gt_12)


def _looks_numeric_date(value):
    if _is_placeholder_text(value):
        return False
    return len(re.findall(r"\d+", str(value))) >= 2


def _parse_text_date(value, date_order):
    if _is_placeholder_text(value) or date_order is None:
        return pd.NaT

    settings_order = "DMY" if date_order == "day_month" else "MDY"
    parsed = dateparser.parse(
        str(value),
        languages=["tr", "en"],
        settings={
            "DATE_ORDER": settings_order,
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    return parsed if parsed is not None else pd.NaT


def _extract_numeric_date_parts(value):
    if _is_placeholder_text(value):
        return None

    tokens = re.findall(r"\d+", str(value))
    if len(tokens) < 2:
        return None

    if len(tokens) >= 3:
        first, second, third = tokens[:3]
        if len(first) == 4 or int(first) > 31:
            return {
                "year": _normalize_year(first),
                "left": int(second),
                "right": int(third),
                "layout": "year_left",
            }
        if len(third) == 4 or int(third) > 31:
            return {
                "year": _normalize_year(third),
                "left": int(first),
                "right": int(second),
                "layout": "year_right",
            }
        if len(third) <= 2:
            return {
                "year": _normalize_year(third),
                "left": int(first),
                "right": int(second),
                "layout": "year_right_two_digit",
            }

    return {
        "year": None,
        "left": int(tokens[0]),
        "right": int(tokens[1]),
        "layout": "no_clear_year",
    }


def _normalize_year(value):
    year = int(value)
    if year < 100:
        return 2000 + year
    return year


def _date_parts_conflict_with_order(parts, date_order):
    left_gt_12 = parts["left"] > 12
    right_gt_12 = parts["right"] > 12
    if left_gt_12 and right_gt_12:
        return True
    if date_order == "day_month" and right_gt_12:
        return True
    if date_order == "month_day" and left_gt_12:
        return True
    return False


def _date_error_example(index, value, reason):
    return {
        "row_index": int(index),
        "value": None if pd.isna(value) else str(value),
        "reason": reason,
    }


def _clean_text_value(value):
    if pd.isna(value):
        return pd.NA

    text = ftfy.fix_text(str(value)).strip()
    return re.sub(r"\s+", " ", text)


def _clean_product_name_value(value):
    text = _clean_text_value(value)
    if _is_placeholder_text(text):
        return pd.NA
    return str(text).lower()


def _is_placeholder_text(value):
    if pd.isna(value):
        return True
    normalized = ftfy.fix_text(str(value)).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in PLACEHOLDER_VALUES


def _placeholder_mask(values):
    normalized = values.astype("string").str.strip().str.lower()
    normalized = normalized.str.replace(r"\s+", " ", regex=True)
    return values.isna() | normalized.isin(PLACEHOLDER_VALUES)


def _normalize_numeric_text(value):
    if _is_placeholder_text(value):
        return pd.NA

    text = str(value).strip().lower()
    text = re.sub(r"(try|tl|usd|eur)", "", text)
    text = re.sub(r"[^0-9,\.\-]", "", text)
    if not re.search(r"\d", text):
        return pd.NA

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    return text


def _parse_price_like_number(value):
    if _is_placeholder_text(value):
        return None

    price = Price.fromstring(str(value))
    if price.amount_float is not None:
        return price.amount_float

    numeric = pd.to_numeric(str(value).replace(",", "."), errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _build_column_report(original, cleaned, column_name):
    original_missing = original.isna()
    cleaned_missing = cleaned.isna()
    failed_mask = ~original_missing & cleaned_missing
    converted_mask = ~cleaned_missing

    return {
        "column_name": column_name,
        "source_dtype": str(original.dtype),
        "target_dtype": str(cleaned.dtype),
        "total_rows": int(len(original)),
        "source_missing_count": int(original_missing.sum()),
        "cleaned_missing_count": int(cleaned_missing.sum()),
        "converted_count": int(converted_mask.sum()),
        "failed_count": int(failed_mask.sum()),
        "failed_examples": _examples_from_mask(original, failed_mask),
        "sample_cleaned_values": [
            None if pd.isna(value) else str(value)
            for value in cleaned.head(10).tolist()
        ],
    }


def _examples_from_mask(series, mask, limit=10):
    examples = []
    for index, value in series[mask].head(limit).items():
        examples.append({
            "row_index": int(index),
            "value": None if pd.isna(value) else str(value),
        })
    return examples


def _safe_scalar(value):
    return None if pd.isna(value) else float(value)


# def _setup_cleaning_logger(log_path):
#     logger = logging.getLogger("data_cleaning")
#     logger.setLevel(logging.DEBUG)
#     logger.propagate = False
#     logger.handlers.clear()

#     handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
#     handler.setLevel(logging.DEBUG)
#     handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
#     logger.addHandler(handler)

#     return logger


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
