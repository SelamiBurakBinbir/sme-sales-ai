import hashlib
import html
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_insights import LLM_NOT_RUNNING_MESSAGE, generate_ai_insights
from column_profiler import profile_dataframe
from data_cleaner import clean_standardized_dataframe
from field_scorer import (
    score_basic_fields,
    score_discount,
    score_revenue,
    validate_revenue_formula,
)
from output_writer import build_standardized_dataframe
from sales_analysis import analyze_cleaned_dataframe, render_sales_analysis
from ml_forecast import build_ml_source_dataframe, run_weekly_forecast


FIELD_LABELS = {
    "date": "Date",
    "product_name": "Product name",
    "quantity": "Quantity",
    "unit_price": "Unit price",
    "revenue": "Revenue",
}

REVENUE_DERIVE_OPTION = "Derive from quantity * unit price"
NO_DISCOUNT_OPTION = "No discount column"
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]
# APP_UPLOADS_DIR = Path(__file__).resolve().parent / "app_uploads"
# LAST_UPLOAD_METADATA_PATH = APP_UPLOADS_DIR / "last_upload.json"
AI_INSIGHTS_PROMPT_VERSION = "gemini_multi_horizon_forecast_v5"
FORECAST_MAX_HORIZON = 12
FORECAST_DISPLAY_HORIZONS = [1, 2, 4, 8, 12]
FORECAST_SCENARIO_ADJUSTMENT_CAP = 0.30
FORECAST_TREND_THRESHOLD = 0.05
PARETO_ABC_HELP_TEXT = (
    "Pareto / ABC Product Analysis ranks products by their revenue contribution "
    "within the selected date range. It helps identify which products generate "
    "most of the total revenue instead of treating all products equally. Products "
    "are sorted from highest to lowest revenue, then their cumulative revenue "
    "share is calculated. Class A products are the main revenue drivers and "
    "usually make up roughly the first 80% of total revenue. Class B products "
    "contribute the next important share, while Class C products have a smaller "
    "individual impact. This analysis is useful for understanding which products "
    "deserve more attention in pricing, stock planning, marketing, and sales "
    "strategy. It is a rule-based business analysis, not a machine learning "
    "forecast."
)
MAE_HELP = (
    "Shows the average weekly prediction error."
)
RMSE_HELP = (
    "Highlights prediction error with more weight on large mistakes."
)
SMAPE_HELP = (
    "Shows prediction error as a percentage."
)
RELATIVE_MAE_HELP = (
    "Shows the average error relative to typical weekly revenue."
)
RELATIVE_RMSE_HELP = (
    "Shows larger forecast errors relative to typical weekly revenue."
)
SELECTED_MODEL_HELP = (
    "Shows the best-performing forecasting model."
)
FINAL_MODEL_FEATURE_COUNT_HELP = (
    "Shows how many input features were used by the final model."
)
SELECTED_COLUMNS_HELP = (
    "Shows how many dataset columns formed the forecast source data before "
    "weekly feature engineering."
)


def main():
    st.set_page_config(
        page_title="Sales Data Analysis",
        layout="wide",
    )
    _inject_styles()
    _init_state()

    _render_app_header()

    upload_tab, mapping_tab, cleaning_tab, analysis_tab, forecast_tab, ai_tab = st.tabs(
        [
            "1. Upload & Preview",
            "2. Column Mapping",
            "3. Cleaning & Standardization",
            "4. Analysis Dashboard",
            "5. ML Forecast",
            "6. AI Insights",
        ]
    )

    with upload_tab:
        _render_upload_preview()

    with mapping_tab:
        _render_column_mapping()

    with cleaning_tab:
        _render_cleaning_standardization()

    with analysis_tab:
        _render_analysis_dashboard()

    with forecast_tab:
        _render_ml_forecast()

    with ai_tab:
        _render_ai_insights()


def _render_app_header():
    uploaded_name = st.session_state.uploaded_name or "No dataset loaded"
    status_items = "".join(
        (
            f'<span class="status-pill {css_class} {"is-done" if is_done else "is-pending"}">'
            f'{html.escape(label)}: {"done" if is_done else "pending"}</span>'
        )
        for label, is_done, css_class in _workspace_status_items()
    )

    st.markdown(
        f"""
        <section class="app-hero">
            <div>
                <h1>Sales Data Analysis</h1>
                <p>An intelligent platform for sales analysis, forecasting, and decision support.</p>
            </div>
            <div class="hero-status">
                <span>Dataset</span>
                <strong>{html.escape(uploaded_name)}</strong>
                <div class="status-pills">{status_items}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _workspace_status_items():
    return [
        ("Uploaded", st.session_state.raw_df is not None, "status-upload"),
        ("Cleaned", st.session_state.cleaning_result is not None, "status-clean"),
        ("Forecasted", st.session_state.forecast_result is not None, "status-forecast"),
        ("AI Insight", bool(st.session_state.ai_insights), "status-ai"),
    ]


def _render_page_intro(title, description, css_class=""):
    class_attr = f" page-intro {html.escape(css_class)}".strip()
    st.markdown(
        f"""
        <div class="{class_attr}">
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_title(title, description=None, help_text=None, css_class=""):
    class_attr = f"section-title {html.escape(css_class)}".strip()
    body = f"<p>{html.escape(description)}</p>" if description else ""
    st.markdown(
        f"""
        <div class="{class_attr}">
            <h3>{html.escape(title)}</h3>
            {body}
        </div>
        """,
        help=help_text,
        unsafe_allow_html=True,
    )


def _init_state():
    defaults = {
        "uploaded_fingerprint": None,
        "uploaded_name": None,
        "raw_df": None,
        "profiles": None,
        "basic_scores": None,
        "revenue_scores": None,
        "discount_scores": None,
        "mapping": None,
        "revenue_validation": None,
        "standardized_df": None,
        "cleaning_result": None,
        "analysis_result": None,
        "analysis_text": None,
        "analysis_range": None,
        "analysis_dataset_key": None,
        "cleaned_dataset_key": None,
#        "uploaded_path": None,
        "uploaded_at": None,
        "ml_source_df": None,
        "ml_extra_columns": None,
        "ml_excluded_columns": None,
        "forecast_result": None,
        "forecast_key": None,
        "ai_insights": None,
        "ai_summary": None,
        "ai_prompt_version": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_downstream_state():
    keys = [
        "raw_df",
        "profiles",
        "basic_scores",
        "revenue_scores",
        "discount_scores",
        "mapping",
        "revenue_validation",
        "standardized_df",
        "cleaning_result",
        "analysis_result",
        "analysis_text",
        "analysis_range",
        "analysis_dataset_key",
        "cleaned_dataset_key",
        "ml_source_df",
        "ml_extra_columns",
        "ml_excluded_columns",
        "forecast_result",
        "forecast_key",
        "ai_insights",
        "ai_summary",
        "ai_prompt_version",
    ]
    for key in keys:
        st.session_state[key] = None

    for key in list(st.session_state.keys()):
        if key.startswith(("mapping_", "analysis_")):
            del st.session_state[key]


def _render_upload_preview():
    _render_page_intro(
        "Upload & Preview",
        "Restore or upload a sales dataset, then verify its structure before mapping columns.",
    )

#    if st.session_state.raw_df is None:
#        _render_load_last_upload()

    with st.container(border=True):
        _render_section_title(
            "Dataset Source",
            "Supported formats: CSV, XLSX, and XLS. Uploaded files are kept only for the current session.",
#            "Supported formats: CSV, XLSX, and XLS. Uploaded files are stored locally for refresh support.",
        )
        uploaded_file = st.file_uploader(
            "Upload a sales dataset",
            type=SUPPORTED_FILE_TYPES,
        )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        fingerprint = (
            uploaded_file.name,
            uploaded_file.size,
            hashlib.sha256(uploaded_bytes).hexdigest(),
        )
        if fingerprint != st.session_state.uploaded_fingerprint:
            _reset_downstream_state()
            st.session_state.uploaded_fingerprint = fingerprint
            st.session_state.uploaded_name = Path(uploaded_file.name).name
            try:
                st.session_state.uploaded_at = _utc_now_iso()
                st.session_state.raw_df = _read_uploaded_dataset(
                    uploaded_file.name,
                    uploaded_bytes,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"The file could not be read: {exc}")
                st.session_state.raw_df = None
            # try:
            #     saved_path = _save_uploaded_file(uploaded_file.name, uploaded_bytes)
            #     st.session_state.raw_df = _read_uploaded_dataset(
            #         uploaded_file.name,
            #         uploaded_bytes,
            #     )
            #     st.session_state.uploaded_path = str(saved_path)
            #     st.session_state.uploaded_at = _utc_now_iso()
            #     _write_last_upload_metadata(
            #         saved_path,
            #         st.session_state.uploaded_name,
            #         st.session_state.uploaded_at,
            #     )
            #     st.rerun()
            # except Exception as exc:
            #     st.error(f"The file could not be read: {exc}")
            #     st.session_state.raw_df = None

    df = st.session_state.raw_df
    if df is None:
        st.info("Upload a sales dataset to begin.")
        return

    st.success(f"{st.session_state.uploaded_name} uploaded.")

    with st.container(border=True):
        _render_section_title("Dataset Snapshot")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Rows", f"{len(df):,}")
        metric_cols[1].metric("Columns", f"{len(df.columns):,}")
        metric_cols[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        # if st.session_state.uploaded_path:
        #     st.caption(f"Stored upload: {st.session_state.uploaded_path}")
        if st.session_state.uploaded_at:
            st.caption(f"Uploaded: {_format_uploaded_at(st.session_state.uploaded_at)}")

    with st.container(border=True):
        _render_section_title("Data Preview", "First 50 rows from the uploaded dataset.")
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    with st.expander("Column summary", expanded=False):
        summary = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(dtype) for dtype in df.dtypes],
                "missing_count": [int(df[column].isna().sum()) for column in df.columns],
                "missing_ratio": [
                    round(float(df[column].isna().mean()), 4) for column in df.columns
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_column_mapping():
    _render_page_intro(
        "Column Mapping",
        "Match the uploaded columns to the standard sales schema before cleaning.",
    )

    df = st.session_state.raw_df
    if df is None:
        st.info("Upload a dataset before mapping columns.")
        return

    _ensure_mapping_artifacts(df)

    mapping = {}
    columns = list(df.columns)
    basic_scores = st.session_state.basic_scores

    with st.container(border=True):
        _render_section_title(
            "Core Schema",
            "The app predicts matches automatically. You can customize any field manually.",
        )
        field_cols = st.columns(4)
        for index, target in enumerate(["date", "product_name", "quantity", "unit_price"]):
            with field_cols[index]:
                mapping[target] = _render_field_selectbox(
                    target,
                    columns,
                    basic_scores.get(target, []),
                )

    with st.container(border=True):
        _render_section_title(
            "Revenue Rules",
            "Discount is optional and is used only to validate revenue consistency.",
        )
        rule_cols = st.columns([1, 2])
        with rule_cols[0]:
            mapping["discount"] = _render_discount_selectbox(columns)
        with rule_cols[1]:
            mapping["revenue"] = _render_revenue_selectbox(df, columns, mapping)
    st.session_state.mapping = mapping

    warnings = _mapping_warnings(mapping, include_revenue_validation=False)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        _render_mapping_status(mapping)

    preview = build_standardized_dataframe(df.head(25), mapping)
    with st.container(border=True):
        _render_section_title(
            "Standardized Preview",
            "A 25-row preview of the schema that will be sent into cleaning.",
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)

    with st.expander("Candidate scores", expanded=False):
        _render_candidate_scores()


def _render_cleaning_standardization():
    _render_page_intro(
        "Cleaning & Standardization",
        "Build the standard schema in memory, clean values, validate quality, and prepare downloadable outputs.",
    )

    df = st.session_state.raw_df
    mapping = st.session_state.mapping
    if df is None:
        st.info("Upload a dataset first.")
        return
    if not mapping:
        st.info("Complete the column mapping before cleaning.")
        return

    warnings = _mapping_warnings(mapping)
    if warnings:
        for warning in warnings:
            st.warning(warning)
        st.stop()

    with st.container(border=True):
        _render_section_title(
            "Cleaning Run",
            "Standardize your sales data, review cleaning results, and download the cleaned outputs when ready.",
        )
        if st.session_state.cleaning_result is None:
            run_clicked = st.button("Standardize and clean", type="primary")
        else:
            run_clicked = False
            st.markdown(
                '<div class="cleaning-ready-message">The latest cleaning result is ready.</div>',
                unsafe_allow_html=True,
            )

    if run_clicked:
        try:
            standardized_df = build_standardized_dataframe(df, mapping)
            discount_series = df[mapping["discount"]] if mapping.get("discount") else None
            cleaning_result = clean_standardized_dataframe(
                standardized_df,
                discount_series=discount_series,
            )
            st.session_state.standardized_df = standardized_df
            st.session_state.cleaning_result = cleaning_result
            ml_source = build_ml_source_dataframe(
                df,
                cleaning_result.dataframe,
                mapping,
            )
            st.session_state.ml_source_df = ml_source["dataframe"]
            st.session_state.ml_extra_columns = ml_source["extra_columns_used"]
            st.session_state.ml_excluded_columns = ml_source["excluded_columns"]
            st.session_state.cleaned_dataset_key = _dataframe_digest(
                cleaning_result.dataframe
            )
            st.session_state.analysis_result = None
            st.session_state.analysis_text = None
            st.session_state.analysis_range = None
            st.session_state.analysis_dataset_key = None
            st.session_state.forecast_result = None
            st.session_state.forecast_key = None
            st.session_state.ai_insights = None
            st.session_state.ai_summary = None
            st.rerun()
        except Exception as exc:
            st.error(f"Standardization or cleaning failed: {exc}")
            return

    cleaning_result = st.session_state.cleaning_result
    if cleaning_result is None:
        st.info("Run standardization and cleaning when the mapping is ready.")
        return

    with st.container(border=True):
        _render_section_title("Cleaning Report")
        _render_cleaning_report(cleaning_result.report)

    with st.container(border=True):
        _render_section_title("Cleaned Data Preview", "First 50 cleaned rows from the standard schema.")
        st.dataframe(cleaning_result.dataframe.head(50), use_container_width=True, hide_index=True)

    with st.container(border=True):
        _render_section_title("Downloads")
        standardized_col, cleaned_col, report_col = st.columns(3)
        standardized_col.download_button(
            "Download standardized CSV",
            data=_dataframe_csv_bytes(st.session_state.standardized_df),
            file_name=_output_file_name("standardized", "csv"),
            mime="text/csv",
            use_container_width=True,
        )
        cleaned_col.download_button(
            "Download cleaned CSV",
            data=_dataframe_csv_bytes(cleaning_result.dataframe),
            file_name=_output_file_name("cleaned", "csv"),
            mime="text/csv",
            use_container_width=True,
        )
        report_col.download_button(
            "Download cleaning report",
            data=json.dumps(cleaning_result.report, ensure_ascii=False, indent=2),
            file_name=_output_file_name("cleaning_report", "json"),
            mime="application/json",
            use_container_width=True,
        )


def _render_analysis_dashboard():
    _render_page_intro(
        "Analysis Dashboard",
        "Explore the cleaned dataset by date range and switch between focused dashboard sections.",
    )

    cleaning_result = st.session_state.cleaning_result
    if cleaning_result is None:
        st.info("Create a cleaned dataset before running analysis.")
        return

    cleaned_df = cleaning_result.dataframe
    date_range = _cleaned_date_range(cleaned_df)
    if date_range is None:
        st.warning("The cleaned dataset does not contain a valid date range.")
        return

    first_date, last_date = date_range
    with st.container(border=True):
        _render_section_title("Analysis Controls")
        control_cols = st.columns([2.2, 0.12, 1.25])
        with control_cols[0]:
            st.markdown(
                '<div class="analysis-control-heading">Dashboard section</div>',
                unsafe_allow_html=True,
            )
            selected_section = st.radio(
                "Dashboard section",
                options=[
                    "Overview",
                    "Product Analysis",
                    "Anomaly Detection",
                    "What-if Simulator",
                    "Reports",
                ],
                horizontal=True,
                key="analysis_dashboard_section",
                label_visibility="collapsed",
            )
        with control_cols[2]:
            st.markdown(
                '<div class="analysis-control-heading is-muted">Date range</div>',
                unsafe_allow_html=True,
            )
            date_cols = st.columns(2)
            start_date = date_cols[0].date_input(
                "Start date",
                value=first_date,
                min_value=first_date,
                max_value=last_date,
                key="analysis_start_date_input",
            )
            end_date = date_cols[1].date_input(
                "End date",
                value=last_date,
                min_value=first_date,
                max_value=last_date,
                key="analysis_end_date_input",
            )

    if start_date > end_date:
        st.error("Start date cannot be later than end date.")
        return

    selected_range = (start_date.isoformat(), end_date.isoformat())
    if selected_section == "Overview":
        try:
            result, _ = _get_analysis_result(cleaned_df, selected_range)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return
        _render_analysis_result(result)
    elif selected_section == "Product Analysis":
        ranged_df = _filter_cleaned_for_range(cleaned_df, *selected_range)
        _render_pareto_abc_analysis(ranged_df)
    elif selected_section == "Anomaly Detection":
        _render_anomaly_detection(cleaned_df, start_date, end_date)
    elif selected_section == "What-if Simulator":
        ranged_df = _filter_cleaned_for_range(cleaned_df, *selected_range)
        _render_what_if_simulator(ranged_df)
    elif selected_section == "Reports":
        try:
            _, analysis_text = _get_analysis_result(cleaned_df, selected_range)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return
        _render_reports_section(analysis_text)


def _read_uploaded_dataset(file_name, data):
    suffix = Path(file_name).suffix.lower()

    if suffix == ".csv":
        return _read_uploaded_csv(data)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(data))

    raise ValueError("Only CSV, XLSX, and XLS files are supported.")


def _read_uploaded_csv(data):
    encodings = ["utf-8", "utf-8-sig", "cp1254", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


# def _render_load_last_upload():
#     metadata = _read_last_upload_metadata()
#     if metadata is None:
#         return

#     upload_path = Path(metadata["path"])
#     if not upload_path.exists():
#         return

#     with st.container(border=True):
#         _render_section_title("Saved Dataset")
#         st.write("A previous uploaded dataset is available.")
#         st.caption(f"File: {metadata['file_name']}")
#         st.caption(f"Uploaded: {_format_uploaded_at(metadata['uploaded_at'])}")
#         if st.button("Load last uploaded dataset"):
#             try:
#                 data = upload_path.read_bytes()
#                 _reset_downstream_state()
#                 st.session_state.uploaded_name = metadata["file_name"]
#                 st.session_state.uploaded_path = str(upload_path)
#                 st.session_state.uploaded_at = metadata["uploaded_at"]
#                 st.session_state.uploaded_fingerprint = (
#                     metadata["file_name"],
#                     upload_path.stat().st_size,
#                     hashlib.sha256(data).hexdigest(),
#                 )
#                 st.session_state.raw_df = _read_uploaded_dataset(
#                     metadata["file_name"],
#                     data,
#                 )
#                 st.rerun()
#             except Exception as exc:
#                 st.error(f"The previous upload could not be loaded: {exc}")


# def _save_uploaded_file(file_name, data):
#     APP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
#     original = Path(file_name)
#     safe_stem = _safe_file_stem(original.stem)
#     digest = hashlib.sha256(data).hexdigest()[:12]
#     path = APP_UPLOADS_DIR / f"{safe_stem}_{digest}{original.suffix.lower()}"
#     path.write_bytes(data)
#     return path


# def _write_last_upload_metadata(path, file_name, uploaded_at):
#     APP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
#     metadata = {
#         "path": str(path),
#         "file_name": file_name,
#         "uploaded_at": uploaded_at,
#     }
#     LAST_UPLOAD_METADATA_PATH.write_text(
#         json.dumps(metadata, ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )


# def _read_last_upload_metadata():
#     if not LAST_UPLOAD_METADATA_PATH.exists():
#         return None
#     try:
#         metadata = json.loads(LAST_UPLOAD_METADATA_PATH.read_text(encoding="utf-8"))
#     except json.JSONDecodeError:
#         return None

#     required_keys = {"path", "file_name", "uploaded_at"}
#     if not required_keys.issubset(metadata):
#         return None
#     return metadata


# def _safe_file_stem(stem):
#     safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
#     return safe.strip("._") or "uploaded_dataset"


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_uploaded_at(value):
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%d %b %Y, %H:%M UTC")
    return parsed.strftime("%d %b %Y, %H:%M")


def _ensure_mapping_artifacts(df):
    if st.session_state.profiles is not None:
        return

    profiles = profile_dataframe(df, sample_size=10)
    st.session_state.profiles = profiles
    st.session_state.basic_scores = score_basic_fields(profiles)


def _render_field_selectbox(target, columns, candidates):
    default_column = _default_candidate(columns, candidates)
    current_value = st.session_state.get(f"mapping_{target}", default_column)
    if current_value not in columns:
        current_value = default_column

    selected = st.selectbox(
        FIELD_LABELS[target],
        options=columns,
        index=columns.index(current_value),
        key=f"mapping_{target}",
    )
    _render_top_candidate_hint(candidates)
    return selected


def _render_revenue_selectbox(df, columns, mapping):
    quantity_column = mapping.get("quantity")
    unit_price_column = mapping.get("unit_price")
    discount_column = mapping.get("discount")
    revenue_scores = score_revenue(
        st.session_state.profiles,
        df,
        selected_quantity=quantity_column,
        selected_unit_price=unit_price_column,
        selected_discount=discount_column,
    )
    st.session_state.revenue_scores = revenue_scores

    options = [REVENUE_DERIVE_OPTION, *columns]
    best_revenue = _default_candidate(columns, revenue_scores)
    default_value = best_revenue or REVENUE_DERIVE_OPTION
    current_value = st.session_state.get("mapping_revenue", default_value)
    if current_value not in options:
        current_value = default_value

    selected = st.selectbox(
        FIELD_LABELS["revenue"],
        options=options,
        index=options.index(current_value),
        key="mapping_revenue",
        help="Select a revenue column, or derive it from quantity and unit price.",
    )
    _render_top_candidate_hint(revenue_scores)

    if selected == REVENUE_DERIVE_OPTION:
        st.session_state.revenue_validation = None
        return None

    validation = validate_revenue_formula(
        df,
        selected,
        quantity_column,
        unit_price_column,
        discount_column,
    )
    st.session_state.revenue_validation = validation
    return selected


def _render_discount_selectbox(columns):
    discount_scores = score_discount(st.session_state.profiles)
    st.session_state.discount_scores = discount_scores

    options = [NO_DISCOUNT_OPTION, *columns]
    best_discount = _default_candidate(columns, discount_scores)
    default_value = best_discount or NO_DISCOUNT_OPTION
    current_value = st.session_state.get("mapping_discount", default_value)
    if current_value not in options:
        current_value = default_value

    selected = st.selectbox(
        "Discount (optional)",
        options=options,
        index=options.index(current_value),
        key="mapping_discount",
        help="Optional. Used only for revenue consistency checks, not added to the core schema.",
    )
    _render_top_candidate_hint(discount_scores)

    if selected == NO_DISCOUNT_OPTION:
        return None
    return selected


def _render_top_candidate_hint(candidates):
    if not candidates:
        st.caption("No automatic candidate.")
        return

    best = candidates[0]
    st.caption(f"Suggested: {best['column']}")


def _render_candidate_scores():
    rows = []
    for target in ["date", "product_name", "quantity", "unit_price"]:
        for candidate in st.session_state.basic_scores.get(target, []):
            rows.append(
                {
                    "target": target,
                    "column": candidate["column"],
                    "score": candidate["score"],
                }
            )
    for candidate in st.session_state.revenue_scores or []:
        rows.append(
            {
                "target": "revenue",
                "column": candidate["column"],
                "score": candidate["score"],
            }
        )
    for candidate in st.session_state.discount_scores or []:
        rows.append(
            {
                "target": "discount",
                "column": candidate["column"],
                "score": candidate["score"],
            }
        )

    if not rows:
        st.write("No candidates found.")
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_mapping_status(mapping):
    validation = st.session_state.revenue_validation

    if mapping.get("revenue") is None:
        status_class = "is-info"
        message = (
            "Revenue will be calculated from quantity * unit price; "
            "The mapping looks ready."
        )
    elif validation and not validation["is_valid"]:
        status_class = "is-warning"
        message = "Revenue validation failed; choose another column or derive revenue."
    else:
        status_class = "is-success"
        message = "Revenue validation passed; The mapping looks ready."

    st.markdown(
        f"""
        <div class="mapping-status {status_class}">
            <span>{html.escape(message)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_cleaning_report(report):
    schema = report.get("schema_validation", {})
    mismatch = report.get("quality_checks", {}).get("revenue_mismatch", {})

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total rows", f"{report.get('total_rows', 0):,}")
    metric_cols[1].metric(
        "Revenue filled",
        f"{report['columns']['revenue'].get('filled_from_quantity_unit_price_count', 0):,}",
    )
    metric_cols[2].metric("Revenue mismatches", f"{mismatch.get('mismatch_count', 0):,}")
    metric_cols[3].metric(
        "Schema validation",
        "Passed" if schema.get("success") else "Has errors",
    )
    st.caption(
        "Revenue consistency formula: "
        f"{mismatch.get('formula_label', 'quantity × unit_price - discount')}"
    )

    rows = []
    for column, column_report in report.get("columns", {}).items():
        rows.append(
            {
                "column": column,
                "converted_count": column_report.get("converted_count", 0),
                "failed_count": column_report.get("failed_count", 0),
                "target_dtype": column_report.get("target_dtype", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_analysis_result(result):
    with st.container(border=True):
        _render_section_title("Performance Overview")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Total revenue", _format_money(result.total_revenue))
        metric_cols[1].metric("Total quantity", _format_number(result.total_quantity))
        metric_cols[2].metric("Product variety", f"{result.total_product_variety:,}")
        metric_cols[3].metric("Average unit price", _format_optional_money(result.average_unit_price))
        metric_cols[4].metric("Rows analyzed", f"{result.row_count:,}")

        avg_cols = st.columns(2)
        avg_cols[0].metric(
            "Average daily revenue",
            _format_optional_money(result.average_daily_revenue_empty_dates_as_zero),
        )
        avg_cols[1].metric(
            "Average on sales days",
            _format_optional_money(result.average_daily_revenue_observed_dates_only),
        )

    daily_revenue = pd.DataFrame(result.daily_revenue)
    daily_quantity = pd.DataFrame(result.daily_quantity)
    chart_cols = st.columns(2)
    if not daily_revenue.empty:
        with chart_cols[0].container(border=True):
            st.markdown("#### Daily revenue")
            st.line_chart(daily_revenue, x="date", y="revenue")
    if not daily_quantity.empty:
        with chart_cols[1].container(border=True):
            st.markdown("#### Daily quantity")
            st.line_chart(daily_quantity, x="date", y="quantity")

    with st.container(border=True):
        _render_section_title("Product Tables")
        product_cols = st.columns(3)
        _render_product_table(
            product_cols[0],
            "Top products by revenue",
            result.top_revenue_products,
        )
        _render_product_table(
            product_cols[1],
            "Top selling products",
            result.top_selling_products,
        )
        _render_product_table(
            product_cols[2],
            "Least selling products",
            result.least_selling_products,
        )


def _get_analysis_result(cleaned_df, selected_range):
    dataset_key = st.session_state.cleaned_dataset_key or _dataframe_digest(cleaned_df)
    cache_key = (dataset_key, selected_range[0], selected_range[1])
    if (
        st.session_state.analysis_result is not None
        and st.session_state.analysis_text is not None
        and st.session_state.analysis_dataset_key == dataset_key
        and st.session_state.analysis_range == selected_range
    ):
        return st.session_state.analysis_result, st.session_state.analysis_text

    result, analysis_text = _compute_analysis_cached(
        cleaned_df,
        selected_range[0],
        selected_range[1],
        cache_key,
    )
    st.session_state.analysis_result = result
    st.session_state.analysis_text = analysis_text
    st.session_state.analysis_range = selected_range
    st.session_state.analysis_dataset_key = dataset_key
    return result, analysis_text


@st.cache_data(show_spinner=False)
def _compute_analysis_cached(cleaned_df, start_date, end_date, cache_key):
    del cache_key
    result = analyze_cleaned_dataframe(cleaned_df, start_date, end_date)
    return result, render_sales_analysis(result)


def _render_reports_section(analysis_text):
    with st.container(border=True):
        _render_section_title("Reports", "Download or inspect the generated text analysis report.")
        st.download_button(
            "Download analysis report",
            data=analysis_text,
            file_name=_output_file_name("analysis", "txt"),
            mime="text/plain",
        )

        with st.expander("Text report", expanded=False):
            st.code(analysis_text, language="text")


def _render_anomaly_detection(cleaned_df, start_date, end_date):
    with st.container(border=True):
        _render_section_title(
            "Statistical Anomaly Detection",
            "Unusual revenue days are detected with the IQR method on daily revenue.",
        )
        control_cols = st.columns(
            [1, 0.5, 1.5],
            vertical_alignment="bottom",
        )
        with control_cols[0]:
            multiplier = st.selectbox(
                "IQR multiplier",
                options=[1.5, 2.0, 2.5, 3.0],
                index=2,
                help=(
                    "The IQR multiplier controls how strict or flexible the "
                    "anomaly detection should be. IQR stands for Interquartile "
                    "Range, which measures the spread of the middle 50% of "
                    "daily revenue values. The system calculates the normal "
                    "revenue range using Q1, Q3, and the selected multiplier. "
                    "A lower multiplier such as 1.5 creates a narrower normal "
                    "range, so more days may be flagged as anomalies. A higher "
                    "multiplier such as 2.5 or 3.0 creates a wider normal "
                    "range, so only more extreme revenue changes are marked as "
                    "unusual. This setting helps detect daily revenue values "
                    "that are unusually high or low compared with the selected "
                    "date range."
                ),
            )
        with control_cols[2]:
            include_low = st.radio(
                "Revenue anomaly scope",
                options=[False, True],
                format_func=(
                    lambda value: "Include unusually low revenue days"
                    if value
                    else "High revenue days only"
                ),
                horizontal=True,
                key="include_low_revenue_days_radio",
            )

        anomaly_result = _detect_daily_revenue_anomalies(
            cleaned_df,
            start_date,
            end_date,
            multiplier=multiplier,
            include_low=include_low,
        )
        anomalies = anomaly_result["anomalies"]
        high_anomalies = anomalies[anomalies["reason"] == "Above normal upper bound"]
        total_days = anomaly_result["total_days"]
        anomaly_ratio = len(anomalies) / total_days if total_days else 0

        if high_anomalies.empty:
            highest_date = "No high anomaly"
            highest_revenue = "No data"
        else:
            highest_row = high_anomalies.sort_values("revenue", ascending=False).iloc[0]
            highest_date = str(highest_row["date"])
            highest_revenue = _format_money(highest_row["revenue"])

        metric_cols = st.columns([4, 3, 3, 3, 3] if include_low else 4)
        metric_cols[0].metric(
            "Anomaly days",
            f"{len(anomalies):,} / {total_days:,} days ({anomaly_ratio:.1%})",
        )
        metric_cols[1].metric("Highest anomaly date", highest_date)
        metric_cols[2].metric("Highest anomaly revenue", highest_revenue)
        metric_cols[3].metric(
            "Normal upper bound",
            _format_money(anomaly_result["upper_bound"]),
        )
        if include_low:
            metric_cols[4].metric(
                "Normal lower bound",
                _format_money(anomaly_result["lower_bound"]),
            )

    _render_anomaly_revenue_chart(anomaly_result["daily"])

    with st.container(border=True):
        _render_section_title("Top 10 Unusual Revenue Days")
        if anomalies.empty:
            st.write("No unusual revenue days found for the selected range.")
            return

        display = anomalies.head(10)[["date", "revenue", "quantity", "reason"]].copy()
        display["revenue"] = display["revenue"].round(2)
        display["quantity"] = display["quantity"].round(2)
        st.dataframe(display, use_container_width=True, hide_index=True)

        full_display = anomalies[["date", "revenue", "quantity", "reason"]].copy()
        full_display["revenue"] = full_display["revenue"].round(2)
        full_display["quantity"] = full_display["quantity"].round(2)
        with st.expander("All unusual revenue days", expanded=False):
            st.dataframe(full_display, use_container_width=True, hide_index=True)

        st.download_button(
            "Download all anomaly days",
            data=_dataframe_csv_bytes(full_display),
            file_name=_output_file_name("anomaly_days", "csv"),
            mime="text/csv",
        )


@st.cache_data(show_spinner=False)
def _detect_daily_revenue_anomalies(
    cleaned_df,
    start_date,
    end_date,
    multiplier=2.5,
    include_low=False,
):
    prepared = cleaned_df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    prepared["revenue"] = pd.to_numeric(prepared["revenue"], errors="coerce")
    prepared["quantity"] = pd.to_numeric(prepared["quantity"], errors="coerce")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    ranged = prepared[prepared["date"].between(start, end, inclusive="both")].copy()

    daily_index = pd.date_range(start=start, end=end, freq="D")
    daily = ranged.groupby("date", dropna=True).agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
    )
    daily = daily.reindex(daily_index, fill_value=0)
    daily.index.name = "date"
    daily = daily.reset_index()

    q1 = daily["revenue"].quantile(0.25)
    q3 = daily["revenue"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    high_mask = daily["revenue"] > upper_bound
    low_mask = daily["revenue"] < lower_bound if include_low else pd.Series(False, index=daily.index)
    anomaly_mask = high_mask | low_mask

    daily["is_anomaly"] = anomaly_mask
    daily["anomaly_revenue"] = daily["revenue"].where(anomaly_mask)
    daily["normal_upper_bound"] = upper_bound
    daily["normal_lower_bound"] = lower_bound

    anomalies = daily[anomaly_mask].copy()
    anomalies["reason"] = ""
    anomalies.loc[high_mask, "reason"] = "Above normal upper bound"
    anomalies.loc[low_mask, "reason"] = "Below normal lower bound"
    anomalies["date"] = anomalies["date"].dt.strftime("%Y-%m-%d")
    anomalies = anomalies.sort_values(["revenue", "date"], ascending=[False, True])

    return {
        "daily": daily,
        "anomalies": anomalies,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "multiplier": multiplier,
        "total_days": int(len(daily)),
    }


def _render_anomaly_revenue_chart(daily):
    chart_data = daily[["date", "revenue", "anomaly_revenue", "normal_upper_bound"]].copy()
    chart_data = chart_data.rename(
        columns={
            "revenue": "Daily revenue",
            "anomaly_revenue": "Anomaly days",
            "normal_upper_bound": "Normal upper bound",
        }
    )
    with st.container(border=True):
        _render_section_title("Daily Revenue With Unusual Days")
        st.line_chart(
            chart_data,
            x="date",
            y=["Daily revenue", "Anomaly days", "Normal upper bound"],
        )


def _render_pareto_abc_analysis(ranged_df):
    abc_table = _build_pareto_abc_table(ranged_df)
    if abc_table.empty:
        st.write("No product revenue data found for the selected range.")
        return

    class_counts = abc_table["abc_class"].value_counts().reindex(["A", "B", "C"], fill_value=0)
    a_count = int(class_counts.loc["A"])
    total_products = int(len(abc_table))

    chart_data = class_counts.rename_axis("abc_class").reset_index(name="product_count")
    summary_col, chart_col = st.columns([1.45, 1])
    with summary_col.container(
        border=True,
        key="pareto_summary_card",
        height="stretch",
    ):
        _render_section_title(
            "Pareto / ABC Product Analysis",
            "Products are ranked by revenue contribution for the selected date range.",
            help_text=PARETO_ABC_HELP_TEXT,
        )
        top_metric_cols = st.columns(2)
        top_metric_cols[0].metric("Total products", f"{total_products:,}")
        top_metric_cols[1].metric("A products", f"{int(class_counts.loc['A']):,}")
        bottom_metric_cols = st.columns(2)
        bottom_metric_cols[0].metric("B products", f"{int(class_counts.loc['B']):,}")
        bottom_metric_cols[1].metric("C products", f"{int(class_counts.loc['C']):,}")
        st.info(f"{a_count:,} products make up roughly 80% of total revenue.")

    with chart_col.container(
        border=True,
        key="abc_distribution_card",
        height="stretch",
    ):
        _render_section_title("A/B/C Distribution")
        st.bar_chart(chart_data, x="abc_class", y="product_count")

    display = abc_table.copy()
    display["revenue"] = display["revenue"].round(2)
    display["revenue_share"] = display["revenue_share"].map(lambda value: f"{value:.2%}")
    display["cumulative_share"] = display["cumulative_share"].map(lambda value: f"{value:.2%}")

    with st.container(border=True):
        _render_section_title("ABC Product Table")
        st.dataframe(
            display[
                [
                    "product_name",
                    "revenue",
                    "revenue_share",
                    "cumulative_share",
                    "abc_class",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


@st.cache_data(show_spinner=False)
def _build_pareto_abc_table(ranged_df):
    prepared = ranged_df.copy()
    prepared["revenue"] = pd.to_numeric(prepared["revenue"], errors="coerce")
    prepared["product_name"] = prepared["product_name"].astype("string")
    prepared = prepared.dropna(subset=["product_name", "revenue"])

    product_revenue = (
        prepared.groupby("product_name", dropna=True)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total_revenue = product_revenue["revenue"].sum()
    if product_revenue.empty or total_revenue <= 0:
        return pd.DataFrame()

    product_revenue["revenue_share"] = product_revenue["revenue"] / total_revenue
    product_revenue["cumulative_share"] = product_revenue["revenue_share"].cumsum()
    cumulative_before = product_revenue["cumulative_share"] - product_revenue["revenue_share"]
    product_revenue["abc_class"] = "C"
    product_revenue.loc[cumulative_before < 0.80, "abc_class"] = "A"
    product_revenue.loc[
        (cumulative_before >= 0.80) & (cumulative_before < 0.95),
        "abc_class",
    ] = "B"

    return product_revenue


def _render_what_if_simulator(ranged_df):
    current_revenue = pd.to_numeric(ranged_df["revenue"], errors="coerce").sum()

    with st.container(border=True):
        _render_section_title(
            "What-if Simulator",
            "Simple decision-support simulation based on price and quantity changes.",
        )
        slider_cols = st.columns(2)
        price_change = slider_cols[0].slider(
            "Price change (%)",
            min_value=-100,
            max_value=100,
            value=0,
            step=1,
        )
        quantity_change = slider_cols[1].slider(
            "Quantity change (%)",
            min_value=-100,
            max_value=100,
            value=0,
            step=1,
        )

        price_factor = 1 + price_change / 100
        quantity_factor = 1 + quantity_change / 100
        simulated_revenue = current_revenue * price_factor * quantity_factor
        revenue_difference = simulated_revenue - current_revenue
        percentage_change = (
            revenue_difference / current_revenue
            if current_revenue
            else 0
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Current revenue", _format_money(current_revenue))
        metric_cols[1].metric("Simulated revenue", _format_money(simulated_revenue))
        metric_cols[2].metric("Revenue difference", _format_money(revenue_difference))
        metric_cols[3].metric("Percentage change", f"{percentage_change:.2%}")


def _render_ml_forecast():
    _render_page_intro(
        "ML Forecast",
        "Forecast upcoming weekly revenue and compare time-aware model performance.",
    )

    cleaning_result = st.session_state.cleaning_result
    ml_source_df = st.session_state.ml_source_df
    if cleaning_result is None or ml_source_df is None:
        st.info("Create a cleaned dataset before running the forecast.")
        return

    dataset_key = st.session_state.cleaned_dataset_key or _dataframe_digest(
        cleaning_result.dataframe
    )
    forecast_key = (dataset_key, FORECAST_MAX_HORIZON)
    has_current_forecast = (
        st.session_state.forecast_result is not None
        and st.session_state.forecast_key == forecast_key
    )
    run_forecast = False
    horizon = 4

    with st.container(border=True):
        _render_section_title(
            "Forecast Controls",
            "This section looks at past weekly sales revenue and forecasts expected revenue for upcoming weeks.",
        )
        control_cols = st.columns([1, 3])
        with control_cols[0]:
            if has_current_forecast:
                horizon = st.selectbox(
                    "Forecast horizon",
                    options=FORECAST_DISPLAY_HORIZONS,
                    index=2,
                    format_func=lambda value: f"{value} weeks",
                )
            else:
                run_forecast = st.button(
                    "Run Forecast",
                    type="primary",
                    key="run_forecast_button",
                )

    if run_forecast or has_current_forecast:
        if run_forecast or st.session_state.forecast_result is None:
            try:
                result = run_weekly_forecast(
                    ml_source_df,
                    st.session_state.ml_extra_columns or [],
                    st.session_state.ml_excluded_columns or [],
                    FORECAST_MAX_HORIZON,
                )
                st.session_state.forecast_result = result
                st.session_state.forecast_key = forecast_key
                st.session_state.ai_insights = None
                st.session_state.ai_summary = None
                st.rerun()
            except ImportError:
                st.error(
                    "scikit-learn is required for ML Forecast. "
                    "Install dependencies from requirements.txt."
                )
                return
            except ValueError as exc:
                st.warning(str(exc))
                return

        _render_forecast_result(st.session_state.forecast_result, horizon)
        return

    st.info("Run the forecast to view horizon options and weekly revenue estimates.")


def _render_forecast_result(result, display_horizon):
    display = _build_forecast_display_summary(result, display_horizon)
    metrics = result["selected_metrics"]
    selected_columns = _forecast_selected_columns(result)

    comparison_col, scenario_col = st.columns([0.9, 1.1])
    with comparison_col.container(border=True, height="stretch"):
        _render_section_title(
            "Previous Period Comparison",
            f"Compares the next {display_horizon} forecast weeks with the latest matching historical period.",
        )
        comparison = display["previous_period_comparison"]
        comparison_metrics = st.columns(2)
        comparison_metrics[0].metric(
            "Previous period revenue",
            _format_optional_money(comparison["previous_period_revenue"]),
        )
        comparison_metrics[1].metric(
            "Forecasted revenue",
            _format_money(comparison["forecasted_revenue"]),
        )
        comparison_delta = st.columns(2)
        comparison_delta[0].metric(
            "Difference",
            _format_optional_money(comparison["difference"]),
        )
        comparison_delta[1].metric(
            "Change percent",
            _format_optional_percent(comparison["change_percent"]),
        )

    with scenario_col.container(border=True, height="stretch"):
        _render_section_title(
            "Forecast Scenarios",
            "A simple expected range based on the selected model error rate.",
        )
        scenarios = display["scenarios"]
        scenario_metrics = st.columns(3)
        scenario_metrics[0].metric(
            "Conservative",
            _format_money(scenarios["conservative"]),
        )
        scenario_metrics[1].metric("Expected", _format_money(scenarios["expected"]))
        scenario_metrics[2].metric(
            "Optimistic",
            _format_money(scenarios["optimistic"]),
        )
        st.caption(_forecast_scenario_caption(metrics, scenarios))
        _render_forecast_trend_message(display["trend"])

    with st.container(border=True):
        _render_section_title(
            "Actual vs Predicted + Future Forecast",
            "Actual revenue is real past weekly revenue. Predicted revenue is the model forecast for the past test period. Future forecast is the estimate for upcoming weeks.",
            css_class="is-single-line",
        )
        st.line_chart(
            display["chart_data"],
            x="week_start",
            y=["actual_revenue", "predicted_revenue", "future_forecast"],
        )

    with st.expander("Future forecast table", expanded=False):
        st.dataframe(display["future_forecast"], use_container_width=True, hide_index=True)

    with st.expander("Forecast summary details", expanded=False):
        st.dataframe(
            _build_forecast_summary_table(result, metrics, selected_columns),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Model Comparison", expanded=False):
        st.caption(
            "The system tries several forecasting methods and automatically selects "
            "the one with the lowest error on past data."
        )
        st.dataframe(result["comparison_table"], use_container_width=True, hide_index=True)

    with st.expander("Selected columns", expanded=False):
        st.caption(
            "These columns formed the forecast source data before weekly feature engineering."
        )
        if selected_columns:
            st.write(", ".join(selected_columns))
        else:
            st.write("No selected columns were available.")

    with st.expander("Excluded columns/features with reasons", expanded=False):
        st.caption(
            "These columns were not included because they were not useful for "
            "forecasting. IDs, customer names, or columns with too many different "
            "values often do not improve predictions."
        )
        excluded = result.get("excluded_columns", [])
        if excluded:
            st.dataframe(pd.DataFrame(excluded), use_container_width=True, hide_index=True)
        else:
            st.write("No extra columns were excluded.")


def _forecast_selected_columns(result):
    ml_source_df = st.session_state.get("ml_source_df")
    if ml_source_df is not None:
        return [str(column) for column in ml_source_df.columns]

    fallback_columns = [
        "date",
        "product_name",
        "quantity",
        "unit_price",
        "revenue",
        *result.get("extra_columns_used", []),
    ]
    return list(dict.fromkeys(str(column) for column in fallback_columns))


def _render_forecast_trend_message(trend):
    direction = trend.get("direction", "not_available")
    container_key = {
        "increasing": "forecast_trend_alert_increasing",
        "stable": "forecast_trend_alert_stable",
        "decreasing": "forecast_trend_alert_decreasing",
        "single_period": "forecast_trend_alert_neutral",
        "not_available": "forecast_trend_alert_neutral",
    }.get(direction, "forecast_trend_alert_neutral")
    summary = trend.get("summary", "Forecast trend is not available yet.")
    with st.container(key=container_key):
        st.info(str(summary))


def _build_forecast_summary_table(result, metrics, selected_columns):
    return pd.DataFrame(
        [
            {
                "Name": "MAE",
                "Value": _format_money(metrics["MAE"]),
                "Description": MAE_HELP,
            },
            {
                "Name": "RMSE",
                "Value": _format_money(metrics["RMSE"]),
                "Description": RMSE_HELP,
            },
            {
                "Name": "sMAPE",
                "Value": f"{metrics['sMAPE']:.2f}%",
                "Description": SMAPE_HELP,
            },
            {
                "Name": "Relative MAE",
                "Value": _format_optional_percent(metrics.get("Relative MAE")),
                "Description": RELATIVE_MAE_HELP,
            },
            {
                "Name": "Relative RMSE",
                "Value": _format_optional_percent(metrics.get("Relative RMSE")),
                "Description": RELATIVE_RMSE_HELP,
            },
            {
                "Name": "Selected model",
                "Value": result["selected_model"],
                "Description": SELECTED_MODEL_HELP,
            },
            {
                "Name": "Final model feature count",
                "Value": f"{result['feature_count']:,}",
                "Description": FINAL_MODEL_FEATURE_COUNT_HELP,
            },
            {
                "Name": "Selected columns",
                "Value": f"{len(selected_columns):,}",
                "Description": SELECTED_COLUMNS_HELP,
            },
        ]
    )


def _build_forecast_display_summary(result, horizon):
    future_forecast = result.get("future_forecast", pd.DataFrame()).head(horizon).copy()
    forecast_values = pd.to_numeric(
        future_forecast.get("future_forecast", pd.Series(dtype=float)),
        errors="coerce",
    )
    forecasted_total = float(forecast_values.sum()) if not forecast_values.empty else 0
    previous_revenue = _previous_period_revenue(result, horizon)
    difference = None
    change_percent = None
    if previous_revenue is not None:
        difference = forecasted_total - previous_revenue
        if previous_revenue != 0:
            change_percent = difference / previous_revenue * 100

    adjustment_ratio = _forecast_scenario_adjustment_ratio(
        result.get("selected_metrics", {}).get("sMAPE")
    )
    scenarios = {
        "conservative": forecasted_total * (1 - adjustment_ratio),
        "expected": forecasted_total,
        "optimistic": forecasted_total * (1 + adjustment_ratio),
        "adjustment_percent": adjustment_ratio * 100,
    }

    return {
        "horizon": horizon,
        "future_forecast": future_forecast,
        "forecasted_total_revenue": forecasted_total,
        "average_forecasted_weekly_revenue": _safe_series_mean(forecast_values),
        "min_forecasted_weekly_revenue": _safe_series_min(forecast_values),
        "max_forecasted_weekly_revenue": _safe_series_max(forecast_values),
        "previous_period_comparison": {
            "previous_period_revenue": previous_revenue,
            "forecasted_revenue": forecasted_total,
            "difference": difference,
            "change_percent": change_percent,
        },
        "scenarios": scenarios,
        "trend": _build_forecast_trend_summary(future_forecast, horizon),
        "chart_data": _slice_forecast_chart_data(result, future_forecast),
    }


def _previous_period_revenue(result, horizon):
    historical = result.get("historical_weekly_revenue")
    if historical is None or historical.empty:
        return None

    prepared = historical.copy()
    prepared["weekly_revenue"] = pd.to_numeric(
        prepared["weekly_revenue"],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=["weekly_revenue"]).sort_values("week_start")
    if len(prepared) < horizon:
        return None
    return float(prepared.tail(horizon)["weekly_revenue"].sum())


def _forecast_scenario_adjustment_ratio(smape):
    if smape is None or pd.isna(smape):
        return 0
    return min(max(float(smape) / 100, 0), FORECAST_SCENARIO_ADJUSTMENT_CAP)


def _forecast_scenario_caption(metrics, scenarios):
    current_adjustment = _format_optional_percent(scenarios["adjustment_percent"])
    smape = metrics.get("sMAPE")
    cap_applied = (
        smape is not None
        and not pd.isna(smape)
        and max(float(smape) / 100, 0) > FORECAST_SCENARIO_ADJUSTMENT_CAP
    )
    suffix = " after cap" if cap_applied else ""
    return (
        "Scenario range uses the selected model error rate. "
        f"Current adjustment: {current_adjustment}{suffix}."
    )


def _build_forecast_trend_summary(future_forecast, horizon):
    values = pd.to_numeric(
        future_forecast.get("future_forecast", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if values.empty:
        return {
            "direction": "not_available",
            "summary": "Forecast trend is not available yet.",
            "first_forecasted_weekly_revenue": None,
            "last_forecasted_weekly_revenue": None,
        }

    first_value = float(values.iloc[0])
    last_value = float(values.iloc[-1])
    if horizon == 1 or len(values) == 1:
        return {
            "direction": "single_period",
            "summary": (
                "Single-period forecast: the next week is expected to reach "
                f"{_format_money(first_value)} in revenue."
            ),
            "first_forecasted_weekly_revenue": first_value,
            "last_forecasted_weekly_revenue": last_value,
        }

    if first_value == 0:
        relative_change = 0 if last_value == 0 else 1
    else:
        relative_change = (last_value - first_value) / abs(first_value)

    if relative_change > FORECAST_TREND_THRESHOLD:
        direction = "increasing"
        summary = (
            f"Forecast trend is increasing across the next {horizon} weeks, "
            f"from {_format_money(first_value)} to {_format_money(last_value)}."
        )
    elif relative_change < -FORECAST_TREND_THRESHOLD:
        direction = "decreasing"
        summary = (
            f"Forecast trend is decreasing across the next {horizon} weeks, "
            f"from {_format_money(first_value)} to {_format_money(last_value)}."
        )
    else:
        direction = "stable"
        summary = (
            f"Forecast trend is stable across the next {horizon} weeks, "
            f"staying near {_format_money(first_value)} to {_format_money(last_value)}."
        )

    return {
        "direction": direction,
        "summary": summary,
        "first_forecasted_weekly_revenue": first_value,
        "last_forecasted_weekly_revenue": last_value,
    }


def _slice_forecast_chart_data(result, future_forecast):
    chart_data = result.get("chart_data", pd.DataFrame()).copy()
    if chart_data.empty or future_forecast.empty:
        return chart_data

    future_weeks = pd.to_datetime(future_forecast["week_start"], errors="coerce")
    chart_weeks = pd.to_datetime(chart_data["week_start"], errors="coerce")
    is_future_row = chart_data["future_forecast"].notna()
    selected_future_row = chart_weeks.isin(future_weeks)
    return chart_data[(~is_future_row) | selected_future_row].copy()


def _safe_series_mean(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _safe_series_min(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.min())


def _safe_series_max(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max())


def _render_ai_insights():
    _render_page_intro(
        "AI Insights",
        "Generate a short business explanation from compact analysis, anomaly, product, and sales summaries, including 1, 2, 4, 8, and 12-week projections if ML Forecast is available.",
        css_class="is-single-line",
    )

    if st.session_state.ai_prompt_version != AI_INSIGHTS_PROMPT_VERSION:
        st.session_state.ai_insights = None
        st.session_state.ai_summary = None
        st.session_state.ai_prompt_version = AI_INSIGHTS_PROMPT_VERSION

    cleaning_result = st.session_state.cleaning_result
    if cleaning_result is None:
        st.info("Create a cleaned dataset before generating AI insights.")
        return

    if not st.session_state.ai_insights:
        with st.container(border=True):
            _render_section_title(
                "Insight Generator",
                "Turn the latest analysis results into a concise business explanation using gemini-3.1-flash-lite's capabilities.",
            )
            if st.button("Generate AI Insights", type="primary"):
                try:
                    summary = _build_ai_summary()
                except ValueError as exc:
                    st.warning(str(exc))
                    return
                st.session_state.ai_summary = summary
                try:
                    st.session_state.ai_insights = generate_ai_insights(summary)
                    st.session_state.ai_prompt_version = AI_INSIGHTS_PROMPT_VERSION
                    st.rerun()
                except RuntimeError as exc:
                    if str(exc) == LLM_NOT_RUNNING_MESSAGE:
                        st.error(LLM_NOT_RUNNING_MESSAGE)
                    else:
                        st.error(LLM_NOT_RUNNING_MESSAGE)
                    return
        st.info("Generate AI insights when the cleaned analysis summary is ready.")

    if st.session_state.ai_insights:
        with st.container(border=True):
            _render_section_title("Business Explanation")
            _render_ai_markdown(st.session_state.ai_insights)

    if st.session_state.ai_summary:
        with st.expander("Data sent to AI", expanded=False):
            st.json(st.session_state.ai_summary)


def _render_ai_markdown(markdown_text):
    st.markdown(_ai_markdown_to_html(markdown_text), unsafe_allow_html=True)


def _ai_markdown_to_html(markdown_text):
    lines = markdown_text.splitlines()
    html_lines = [
        """
        <style>
        .ai-insights-markdown strong {
            color: #f59e0b;
            font-weight: 800;
        }
        .ai-insights-markdown em {
            color: #38bdf8;
            font-style: italic;
            font-weight: 650;
        }
        </style>
        """,
        "<div class=\"ai-insights-markdown\">",
    ]
    list_type = None

    def close_list():
        nonlocal list_type
        if list_type:
            html_lines.append(f"</{list_type}>")
            list_type = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            close_list()
            level = min(len(heading_match.group(1)), 4)
            html_lines.append(
                f"<h{level}>{_ai_inline_markdown_to_html(heading_match.group(2))}</h{level}>"
            )
            continue

        unordered_match = re.match(r"^[-*]\s+(.+)$", stripped)
        if unordered_match:
            if list_type != "ul":
                close_list()
                html_lines.append("<ul>")
                list_type = "ul"
            html_lines.append(
                f"<li>{_ai_inline_markdown_to_html(unordered_match.group(1))}</li>"
            )
            continue

        ordered_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ordered_match:
            if list_type != "ol":
                close_list()
                html_lines.append("<ol>")
                list_type = "ol"
            html_lines.append(
                f"<li>{_ai_inline_markdown_to_html(ordered_match.group(1))}</li>"
            )
            continue

        close_list()
        html_lines.append(f"<p>{_ai_inline_markdown_to_html(stripped)}</p>")

    close_list()
    html_lines.append("</div>")
    return "\n".join(html_lines)


def _ai_inline_markdown_to_html(text):
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"`([^`\n]+)`", r"\1", escaped)
    return escaped


def _build_ai_summary():
    cleaned_df = st.session_state.cleaning_result.dataframe
    date_range = _cleaned_date_range(cleaned_df)
    if date_range is None:
        raise ValueError("The cleaned dataset does not contain a valid date range.")

    selected_range = st.session_state.analysis_range
    if selected_range is None:
        selected_range = (date_range[0].isoformat(), date_range[1].isoformat())

    result, _ = _get_analysis_result(cleaned_df, selected_range)
    ranged_df = _filter_cleaned_for_range(cleaned_df, *selected_range)
    abc_table = _build_pareto_abc_table(ranged_df)
    anomaly_summary = _build_ai_anomaly_summary(cleaned_df, selected_range)
    pareto_summary = _build_ai_pareto_summary(abc_table)

    summary = {
        "date_range": {
            "start": selected_range[0],
            "end": selected_range[1],
        },
        "kpi_summary": {
            "total_revenue": result.total_revenue,
            "total_quantity": result.total_quantity,
            "product_variety": result.total_product_variety,
            "average_unit_price": result.average_unit_price,
            "average_daily_revenue": result.average_daily_revenue_empty_dates_as_zero,
            "rows_analyzed": result.row_count,
        },
        "top_products_summary": {
            "top_by_revenue": result.top_revenue_products[:5],
            "top_by_quantity_count": result.top_selling_products[:5],
            "least_selling": result.least_selling_products[:5],
        },
        "pareto_abc_summary": pareto_summary,
        "anomaly_summary": anomaly_summary,
        "ml_forecast_summary": _build_ai_forecast_summary(),
    }
    return _json_ready(summary)


def _build_ai_anomaly_summary(cleaned_df, selected_range):
    anomaly_result = _detect_daily_revenue_anomalies(
        cleaned_df,
        selected_range[0],
        selected_range[1],
        multiplier=2.5,
        include_low=False,
    )
    anomalies = anomaly_result["anomalies"]
    top_anomalies = anomalies.head(5)[["date", "revenue", "quantity", "reason"]]
    highest = None
    if not anomalies.empty:
        highest_row = anomalies.sort_values("revenue", ascending=False).iloc[0]
        highest = {
            "date": str(highest_row["date"]),
            "revenue": float(highest_row["revenue"]),
            "quantity": float(highest_row["quantity"]),
            "reason": str(highest_row["reason"]),
        }

    return {
        "method": "IQR unusual high revenue days",
        "iqr_multiplier": 2.5,
        "anomaly_days": int(len(anomalies)),
        "total_days": int(anomaly_result["total_days"]),
        "normal_upper_bound": float(anomaly_result["upper_bound"]),
        "highest_anomaly": highest,
        "top_anomaly_days": top_anomalies.to_dict(orient="records"),
    }


def _build_ai_pareto_summary(abc_table):
    if abc_table.empty:
        return {
            "total_products": 0,
            "class_counts": {"A": 0, "B": 0, "C": 0},
            "a_products_make_roughly_80_percent": 0,
            "top_a_products": [],
        }

    class_counts = abc_table["abc_class"].value_counts().reindex(["A", "B", "C"], fill_value=0)
    top_a = abc_table[abc_table["abc_class"] == "A"].head(5)
    return {
        "total_products": int(len(abc_table)),
        "class_counts": {
            "A": int(class_counts.loc["A"]),
            "B": int(class_counts.loc["B"]),
            "C": int(class_counts.loc["C"]),
        },
        "a_products_make_roughly_80_percent": int(class_counts.loc["A"]),
        "top_a_products": top_a[
            ["product_name", "revenue", "revenue_share", "cumulative_share", "abc_class"]
        ].to_dict(orient="records"),
    }


def _build_ai_forecast_summary():
    forecast = st.session_state.forecast_result
    if not forecast:
        return {
            "status": "not_run",
            "message": "ML Forecast has not been run yet.",
        }

    metrics = forecast["selected_metrics"]
    horizon_summaries = {}
    for horizon in FORECAST_DISPLAY_HORIZONS:
        display = _build_forecast_display_summary(forecast, horizon)
        future_values = pd.to_numeric(
            display["future_forecast"].get("future_forecast", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        comparison = display["previous_period_comparison"]
        scenarios = display["scenarios"]
        trend = display["trend"]
        horizon_summaries[f"{horizon}_weeks"] = {
            "forecast_period_count": int(len(display["future_forecast"])),
            "forecasted_total_revenue": display["forecasted_total_revenue"],
            "average_forecasted_weekly_revenue": display[
                "average_forecasted_weekly_revenue"
            ],
            "min_forecasted_weekly_revenue": display["min_forecasted_weekly_revenue"],
            "max_forecasted_weekly_revenue": display["max_forecasted_weekly_revenue"],
            "first_forecasted_weekly_revenue": (
                float(future_values.iloc[0]) if not future_values.empty else None
            ),
            "last_forecasted_weekly_revenue": (
                float(future_values.iloc[-1]) if not future_values.empty else None
            ),
            "previous_period_revenue": comparison["previous_period_revenue"],
            "forecast_vs_previous_period_difference": comparison["difference"],
            "forecast_vs_previous_period_change_percent": comparison["change_percent"],
            "conservative_forecast": scenarios["conservative"],
            "expected_forecast": scenarios["expected"],
            "optimistic_forecast": scenarios["optimistic"],
            "scenario_adjustment_percent": scenarios["adjustment_percent"],
            "forecast_trend_direction": trend["direction"],
            "forecast_trend_summary": trend["summary"],
        }

    return {
        "status": "available",
        "target": "weekly revenue",
        "forecast_granularity": "weekly",
        "selected_model": forecast["selected_model"],
        "compared_models": forecast.get("compared_models", []),
        "metrics": {
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "sMAPE": metrics["sMAPE"],
            "Relative MAE": metrics.get("Relative MAE"),
            "Relative RMSE": metrics.get("Relative RMSE"),
        },
        "multi_horizon_forecast_summary": horizon_summaries,
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if pd.isna(value):
        return None
    return value


def _render_product_table(container, title, rows):
    container.markdown(f"#### {title}")
    if not rows:
        container.write("No data.")
        return
    container.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _mapping_warnings(mapping, include_revenue_validation=True):
    warnings = []
    required_targets = ["date", "product_name", "quantity", "unit_price"]
    missing = [target for target in required_targets if not mapping.get(target)]
    if missing:
        warnings.append(f"Missing mapping: {', '.join(missing)}")

    selected_columns = [
        column for target, column in mapping.items() if column is not None and target != "revenue"
    ]
    duplicate_columns = sorted(
        {column for column in selected_columns if selected_columns.count(column) > 1}
    )
    if duplicate_columns:
        warnings.append(
            "The same source column is mapped to multiple required fields: "
            + ", ".join(duplicate_columns)
        )

    revenue = mapping.get("revenue")
    if include_revenue_validation and revenue is not None:
        validation = st.session_state.revenue_validation
        if validation and not validation["is_valid"]:
            warnings.append(
                "Revenue validation failed; choose another column or derive revenue."
            )

    return warnings


def _default_candidate(columns, candidates):
    if candidates:
        candidate_column = candidates[0]["column"]
        if candidate_column in columns:
            return candidate_column
    return columns[0] if columns else None


def _cleaned_date_range(df):
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().date(), dates.max().date()


@st.cache_data(show_spinner=False)
def _filter_cleaned_for_range(df, start_date, end_date):
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return prepared[prepared["date"].between(start, end, inclusive="both")].copy()


def _dataframe_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def _dataframe_digest(df):
    return hashlib.sha256(_dataframe_csv_bytes(df)).hexdigest()


def _output_file_name(kind, extension):
    source_name = st.session_state.uploaded_name or "sales_data"
    stem = Path(source_name).stem or "sales_data"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return f"{safe_stem}_{kind}.{extension}"


def _format_money(value):
    return f"{float(value):,.2f}"


def _format_optional_money(value):
    if value is None or pd.isna(value):
        return "No data"
    return _format_money(value)


def _format_optional_percent(value):
    if value is None or pd.isna(value):
        return "No data"
    return f"{float(value):,.2f}%"


def _format_number(value):
    return f"{float(value):,.2f}"


def _inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0d1117;
            --panel-bg: #151922;
            --panel-bg-2: #1b2130;
            --panel-border: #2a3242;
            --text-main: #f8fafc;
            --text-muted: #a7b0bf;
            --accent: #d93649;
            --accent-soft: rgba(217, 54, 73, 0.18);
        }
        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(180deg, rgba(21, 25, 34, 0.96) 0%, rgba(13, 17, 23, 1) 320px),
                var(--app-bg);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 0.75rem;
            padding-bottom: 4rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        .app-hero {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 2rem;
            padding: 0.75rem 0 1.65rem;
            border-bottom: 1px solid var(--panel-border);
            margin-bottom: 1.1rem;
        }
        .app-hero h1 {
            font-size: 3rem;
            line-height: 1.02;
            margin: 0.25rem 0 0.65rem;
            letter-spacing: 0;
        }
        .app-hero p {
            color: var(--text-muted);
            font-size: 1.02rem;
            margin: 0;
            max-width: 740px;
        }
        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-status {
            min-width: 300px;
            border: 1px solid var(--panel-border);
            background: rgba(21, 25, 34, 0.72);
            border-radius: 8px;
            padding: 1rem 1.1rem;
        }
        .hero-status span {
            display: block;
            color: var(--text-muted);
            font-size: 0.82rem;
        }
        .hero-status strong {
            display: block;
            color: var(--text-main);
            font-size: 1rem;
            margin: 0.18rem 0 0.35rem;
            overflow-wrap: anywhere;
        }
        .status-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.75rem;
        }
        .status-pill {
            border: 1px solid var(--panel-border);
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
            min-width: 8.9rem;
            padding: 0.42rem 0.62rem;
            font-size: 0.76rem;
            font-weight: 800;
            text-align: center;
        }
        .status-pill.is-pending {
            filter: saturate(0.65);
            opacity: 0.68;
        }
        .hero-status .status-pill.is-done {
            opacity: 1;
            filter: none;
        }
        .hero-status .status-upload {
            background: rgba(56, 189, 248, 0.16);
            color: #ffffff !important;
            font-weight: 800 !important;
        }
        .hero-status .status-clean {
            background: rgba(34, 197, 94, 0.16);
            color: #86efac !important;
            font-weight: 800 !important;
        }
        .hero-status .status-forecast {
            background: rgba(245, 158, 11, 0.16);
            color: #facc15 !important;
            font-weight: 800 !important;
            min-width: 10rem;
        }
        .hero-status .status-ai {
            background: rgba(168, 85, 247, 0.16);
            color: #d8b4fe !important;
            font-weight: 800 !important;
        }
        .page-intro {
            margin: 0.9rem 0 1.15rem;
            padding-bottom: 0.9rem;
            border-bottom: 1px solid var(--panel-border);
        }
        .page-intro h2 {
            font-size: 1.9rem;
            line-height: 1.15;
            margin: 0.1rem 0 0.25rem;
            letter-spacing: 0;
        }
        .page-intro p,
        .section-title p {
            color: var(--text-muted);
            font-size: 0.98rem;
            margin: 0;
            max-width: 980px;
        }
        .page-intro.is-single-line p {
            max-width: none;
            white-space: nowrap;
        }
        .section-title.is-single-line p {
            max-width: none;
            white-space: nowrap;
        }
        .section-title {
            margin: 0.3rem 0 1rem;
        }
        .section-title h3 {
            font-size: 1.35rem;
            line-height: 1.22;
            margin: 0 0 0.35rem;
            letter-spacing: 0;
        }
        h1, h2, h3, h4 {
            letter-spacing: 0;
        }
        div[data-testid="stTabs"] {
            margin-top: 0.25rem;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            gap: 0.25rem;
            border-bottom: 1px solid var(--panel-border);
        }
        div[data-testid="stTabs"] button {
            color: var(--text-muted);
            white-space: nowrap;
            padding: 0.85rem 0.95rem 0.9rem;
            border-radius: 6px 6px 0 0;
            transition: background 120ms ease, color 120ms ease;
        }
        div[data-testid="stTabs"] button:hover {
            background: rgba(255, 255, 255, 0.035);
            color: var(--text-main);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #ffffff;
            background: var(--accent-soft);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: #ffffff;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--panel-border);
            background: rgba(21, 25, 34, 0.62);
            border-radius: 8px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.12);
        }
        [data-testid="stMetric"] {
            background: rgba(27, 33, 48, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 94px;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 700;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem;
            color: var(--text-main);
        }
        div[data-testid="stRadio"] [role="radiogroup"] {
            align-items: center !important;
            min-height: 3.15rem;
        }
        .analysis-control-heading {
            color: var(--text-main);
            font-size: 0.92rem;
            font-weight: 800;
            margin: 0 0 0.55rem;
        }
        .analysis-control-heading.is-muted {
            color: var(--text-muted);
        }
        .st-key-analysis_dashboard_section {
            background: rgba(217, 54, 73, 0.08);
            border: 1px solid rgba(217, 54, 73, 0.22);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 0.68rem 0.7rem;
        }
        .st-key-analysis_dashboard_section [data-testid="stMarkdownContainer"] {
            width: auto;
        }
        .st-key-analysis_dashboard_section [role="radiogroup"] {
            gap: 0.28rem;
            justify-content: flex-start;
            width: 100%;
        }
        .st-key-analysis_dashboard_section label[data-baseweb="radio"] {
            background: rgba(15, 20, 30, 0.62);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 7px;
            min-height: 2.5rem;
            padding: 0.4rem 0.52rem;
            transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
        }
        .st-key-analysis_dashboard_section label[data-baseweb="radio"]:hover {
            background: rgba(217, 54, 73, 0.13);
            border-color: rgba(217, 54, 73, 0.36);
        }
        .st-key-analysis_dashboard_section label[data-baseweb="radio"]:has(input:checked) {
            background: rgba(217, 54, 73, 0.22);
            border-color: rgba(217, 54, 73, 0.68);
        }
        label[data-baseweb="radio"] {
            align-items: center !important;
            display: inline-flex !important;
        }
        label[data-baseweb="radio"] > *,
        label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] {
            align-items: center !important;
            display: inline-flex !important;
        }
        label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {
            line-height: normal !important;
            margin: 0 !important;
        }
        .cleaning-ready-message {
            align-items: center;
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.28);
            border-radius: 7px;
            box-sizing: border-box;
            color: #86efac;
            display: flex;
            font-size: 0.95rem;
            font-weight: 700;
            height: 4rem;
            margin-bottom: 1.25rem;
            padding: 0 1.35rem;
            width: 100%;
        }
        .mapping-status {
            align-items: center;
            border-radius: 7px;
            box-sizing: border-box;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.05rem 0 1.25rem;
            min-height: 3.15rem;
            padding: 0.75rem 1.15rem;
            width: 100%;
        }
        .mapping-status.is-success {
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.22);
            color: #86efac;
        }
        .mapping-status.is-warning {
            background: rgba(245, 158, 11, 0.18);
            border: 1px solid rgba(245, 158, 11, 0.24);
            color: #fde68a;
        }
        .mapping-status.is-info {
            background: rgba(56, 189, 248, 0.16);
            border: 1px solid rgba(56, 189, 248, 0.24);
            color: #7dd3fc;
        }
        .mapping-status-formula {
            color: inherit;
            flex-shrink: 0;
            opacity: 0.9;
            text-align: right;
        }
        .st-key-forecast_trend_alert_increasing [data-testid="stAlert"] {
            background: rgba(22, 163, 74, 0.28);
            border: 1px solid rgba(34, 197, 94, 0.62);
        }
        .st-key-forecast_trend_alert_increasing [data-testid="stAlert"] * {
            color: #bbf7d0;
        }
        .st-key-forecast_trend_alert_stable [data-testid="stAlert"] {
            background: rgba(56, 189, 248, 0.16);
            border: 1px solid rgba(56, 189, 248, 0.28);
        }
        .st-key-forecast_trend_alert_stable [data-testid="stAlert"] * {
            color: #7dd3fc;
        }
        .st-key-forecast_trend_alert_decreasing [data-testid="stAlert"] {
            background: rgba(220, 38, 38, 0.28);
            border: 1px solid rgba(248, 113, 113, 0.64);
        }
        .st-key-forecast_trend_alert_decreasing [data-testid="stAlert"] * {
            color: #fecaca;
        }
        .st-key-forecast_trend_alert_neutral [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--panel-border);
        }
        .st-key-forecast_trend_alert_neutral [data-testid="stAlert"] * {
            color: var(--text-muted);
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px !important;
            min-height: 2.75rem;
            font-weight: 750;
            border-color: #3a4457;
            background: rgba(15, 20, 30, 0.55);
            overflow: hidden;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] *,
        .stDownloadButton > button[kind="primary"] *,
        button[data-testid="stBaseButton-primary"] * {
            color: #ffffff !important;
        }
        button[data-testid="stBaseButton-primary"] p,
        button[data-testid="stBaseButton-primary"] span,
        button[data-testid="stBaseButton-primary"] div {
            color: #ffffff !important;
            font-weight: 800 !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--accent);
            color: #ffffff;
        }
        .st-key-run_forecast_button {
            padding-top: 0.15rem;
        }
        .st-key-run_forecast_button button {
            height: 3.15rem;
            min-height: 3.15rem;
        }
        [data-testid="stFileUploader"] {
            background: rgba(27, 33, 48, 0.72);
            border: 1px dashed #3a4457;
            border-radius: 8px;
            overflow: hidden;
            padding: 0.75rem;
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 8px !important;
            overflow: hidden;
        }
        [data-testid="stFileUploader"] section > div,
        [data-testid="stFileUploaderDropzone"] > div {
            border-radius: 8px !important;
            overflow: hidden;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] [role="button"],
        [data-testid="stFileUploader"] button > div {
            border-radius: 8px !important;
            overflow: hidden;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            overflow: hidden;
        }
        details {
            border-color: var(--panel-border) !important;
            border-radius: 8px !important;
            background: rgba(15, 20, 30, 0.42) !important;
        }
        .stAlert {
            border-radius: 8px;
        }
        hr {
            border-color: var(--panel-border);
        }
        .block-container {
            padding-top: 3rem !important;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .app-hero {
                align-items: flex-start;
                flex-direction: column;
            }
            .app-hero h1 {
                font-size: 2.35rem;
            }
            .hero-status {
                min-width: 0;
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
