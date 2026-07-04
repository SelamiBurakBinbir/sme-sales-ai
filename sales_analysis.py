from dataclasses import dataclass

import pandas as pd

from field_scorer import TARGET_FIELDS


@dataclass(frozen=True)
class SalesAnalysisResult:
    start_date: str
    end_date: str
    row_count: int
    total_revenue: float
    total_quantity: float
    total_product_variety: int
    average_unit_price: float | None
    average_daily_revenue_empty_dates_as_zero: float | None
    average_daily_revenue_observed_dates_only: float | None
    top_revenue_products: list[dict]
    top_selling_products: list[dict]
    least_selling_products: list[dict]
    daily_revenue: list[dict]
    daily_quantity: list[dict]


def analyze_cleaned_dataframe(df, start_date, end_date):
    _validate_columns(df)
    start, end = _parse_date_range(start_date, end_date)

    prepared = _prepare_cleaned_dataframe(df)
    ranged = _filter_date_range(prepared, start, end)

    daily = _build_daily_frame(ranged, start, end)
    observed_daily_revenue = daily.loc[daily["has_sales"], "revenue"]

    return SalesAnalysisResult(
        start_date=_format_date(start),
        end_date=_format_date(end),
        row_count=int(len(ranged)),
        total_revenue=_round_money(ranged["revenue"].sum()),
        total_quantity=_round_quantity(ranged["quantity"].sum()),
        total_product_variety=int(ranged["product_name"].nunique(dropna=True)),
        average_unit_price=_round_money_or_none(ranged["unit_price"].mean()),
        average_daily_revenue_empty_dates_as_zero=_round_money_or_none(
            daily["revenue"].mean()
        ),
        average_daily_revenue_observed_dates_only=_round_money_or_none(
            observed_daily_revenue.mean()
        ),
        top_revenue_products=_top_revenue_products(ranged),
        top_selling_products=_product_counts(ranged, ascending=False),
        least_selling_products=_product_counts(ranged, ascending=True),
        daily_revenue=_daily_records(daily, "revenue", _round_money),
        daily_quantity=_daily_records(daily, "quantity", _round_quantity),
    )


def render_sales_analysis(result):
    lines = [
        "Analysis results",
        f"Date range: {result.start_date} - {result.end_date}",
        f"Rows included in analysis: {result.row_count}",
        f"Total revenue: {_format_number(result.total_revenue)}",
        f"Total quantity sold: {_format_number(result.total_quantity)}",
        f"Product variety: {result.total_product_variety}",
        f"Average unit price: {_format_optional(result.average_unit_price)}",
        (
            "Average daily revenue (empty dates counted as zero): "
            f"{_format_optional(result.average_daily_revenue_empty_dates_as_zero)}"
        ),
        (
            "Average daily revenue (sales days only): "
            f"{_format_optional(result.average_daily_revenue_observed_dates_only)}"
        ),
    ]

    lines.extend(_render_table(
        "Top products by revenue",
        result.top_revenue_products,
        ["product_name", "revenue"],
        ["Product", "Revenue"],
    ))
    lines.extend(_render_table(
        "Top selling products",
        result.top_selling_products,
        ["product_name", "count"],
        ["Product", "Count"],
    ))
    lines.extend(_render_table(
        "Least selling products",
        result.least_selling_products,
        ["product_name", "count"],
        ["Product", "Count"],
    ))
    lines.extend(_render_table(
        "Daily revenue",
        result.daily_revenue,
        ["date", "revenue"],
        ["Date", "Revenue"],
    ))
    lines.extend(_render_table(
        "Daily quantity",
        result.daily_quantity,
        ["date", "quantity"],
        ["Date", "Quantity"],
    ))

    return "\n".join(lines)


def _validate_columns(df):
    missing_columns = [column for column in TARGET_FIELDS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"The cleaned dataset is missing columns: {missing_columns}")


def _parse_date_range(start_date, end_date):
    start = pd.to_datetime(start_date, errors="coerce").normalize()
    end = pd.to_datetime(end_date, errors="coerce").normalize()

    if pd.isna(start):
        raise ValueError(f"Start date could not be parsed: {start_date}")
    if pd.isna(end):
        raise ValueError(f"End date could not be parsed: {end_date}")
    if start > end:
        raise ValueError("Start date cannot be later than end date.")

    return start, end


def _prepare_cleaned_dataframe(df):
    prepared = df[TARGET_FIELDS].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()

    for column in ["quantity", "unit_price", "revenue"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["product_name"] = prepared["product_name"].astype("string")
    return prepared


def _filter_date_range(df, start, end):
    return df[df["date"].between(start, end, inclusive="both")].copy()


def _build_daily_frame(df, start, end):
    date_index = pd.date_range(start=start, end=end, freq="D")
    daily = df.groupby("date", dropna=True).agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
        row_count=("date", "size"),
    )
    daily = daily.reindex(date_index, fill_value=0)
    daily.index.name = "date"
    daily["has_sales"] = daily["row_count"] > 0
    return daily.reset_index()


def _top_revenue_products(df, limit=10):
    grouped = (
        df.dropna(subset=["product_name"])
        .groupby("product_name", dropna=True)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
    )
    return [
        {"product_name": str(product_name), "revenue": _round_money(revenue)}
        for product_name, revenue in grouped.items()
    ]


def _product_counts(df, ascending, limit=10):
    counts = (
        df["product_name"]
        .dropna()
        .value_counts(ascending=ascending)
        .head(limit)
    )
    return [
        {"product_name": str(product_name), "count": int(count)}
        for product_name, count in counts.items()
    ]


def _daily_records(daily, value_column, formatter):
    return [
        {
            "date": _format_date(row.date),
            value_column: formatter(getattr(row, value_column)),
        }
        for row in daily.itertuples(index=False)
    ]


def _render_table(title, rows, keys, headers):
    lines = ["", f"{title}:"]
    if not rows:
        lines.append("  No data.")
        return lines

    formatted_rows = [
        [_format_cell(row[key]) for key in keys]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in formatted_rows))
        for index, header in enumerate(headers)
    ]
    alignments = [_column_alignment(key) for key in keys]

    header_cells = [
        _align_cell(header, width, alignment)
        for header, width, alignment in zip(headers, widths, alignments)
    ]
    lines.append("  " + " | ".join(header_cells))
    lines.append("  " + "-+-".join("-" * width for width in widths))

    for row in formatted_rows:
        cells = [
            _align_cell(value, width, alignment)
            for value, width, alignment in zip(row, widths, alignments)
        ]
        lines.append("  " + " | ".join(cells))

    return lines


def _column_alignment(key):
    if key in {"count", "quantity", "revenue"}:
        return "right"
    return "left"


def _align_cell(value, width, alignment):
    if alignment == "right":
        return value.rjust(width)
    return value.ljust(width)


def _format_cell(value):
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_optional(value):
    if value is None:
        return "No data"
    return _format_number(value)


def _format_number(value):
    if value is None:
        return "No data"
    return f"{value:,.2f}"


def _format_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _round_money(value):
    return round(float(value), 2)


def _round_quantity(value):
    return round(float(value), 2)


def _round_money_or_none(value):
    if pd.isna(value):
        return None
    return _round_money(value)
