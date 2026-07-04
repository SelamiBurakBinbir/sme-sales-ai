import numpy as np
import pandas as pd


IDENTIFIER_KEYWORDS = {
    "id",
    "row",
    "no",
    "number",
    "code",
    "postal",
    "zip",
    "address",
    "phone",
    "email",
}
NUMERIC_MEASUREMENT_KEYWORDS = {
    "profit",
    "cost",
    "price",
    "amount",
    "shipping",
    "ship",
    "sales",
    "tax",
    "margin",
    "discount",
    "disc",
    "indirim",
    "fee",
    "total",
    "weight",
    "quantity",
    "qty",
}
DATE_KEYWORDS = {"date", "time", "ship", "delivery", "delivered"}
HIGH_CARDINALITY_UNIQUE_LIMIT = 100
HIGH_CARDINALITY_RATIO_LIMIT = 0.5
PROFILE_MIN_NON_MISSING_RATIO = 0.2
LOW_VARIANCE_UNIQUE_LIMIT = 1


def build_ml_source_dataframe(raw_df, cleaned_df, mapping):
    ml_df = cleaned_df.reset_index(drop=True).copy()
    core_source_columns = {
        mapping.get("date"),
        mapping.get("product_name"),
        mapping.get("quantity"),
        mapping.get("unit_price"),
        mapping.get("revenue"),
    }
    core_source_columns.discard(None)

    selected_columns = []
    excluded_columns = []

    for column in raw_df.columns:
        if column in core_source_columns:
            continue

        profile = _profile_extra_column(raw_df[column], column)
        role, reason = _infer_column_role(
            profile,
            selected_discount_column=mapping.get("discount"),
            column=column,
        )
        if role in {"identifier", "excluded"}:
            excluded_columns.append({
                "column": str(column),
                "role": role,
                "reason": reason,
            })
            continue

        if role == "secondary_date":
            feature_name = _date_delta_feature_name(column)
            feature_name = _safe_extra_column_name(feature_name, ml_df.columns)
            source_dates = pd.to_datetime(raw_df[column], errors="coerce")
            core_dates = pd.to_datetime(ml_df["date"], errors="coerce")
            ml_df[feature_name] = (source_dates - core_dates).dt.days
            selected_columns.append(feature_name)
            continue

        ml_column = _safe_extra_column_name(column, ml_df.columns)
        ml_df[ml_column] = raw_df[column].reset_index(drop=True)
        selected_columns.append(ml_column)

    discount_column = mapping.get("discount")
    if discount_column and discount_column in raw_df.columns and discount_column not in core_source_columns:
        if str(discount_column) not in [str(column) for column in selected_columns]:
            ml_column = _safe_extra_column_name(discount_column, ml_df.columns)
            if ml_column not in ml_df.columns:
                ml_df[ml_column] = raw_df[discount_column].reset_index(drop=True)
            if ml_column not in selected_columns:
                selected_columns.append(ml_column)

    return {
        "dataframe": ml_df,
        "extra_columns_used": selected_columns,
        "excluded_columns": excluded_columns,
    }


def run_weekly_forecast(ml_source_df, extra_columns, excluded_columns, horizon):
    sklearn = _load_sklearn()
    weekly, feature_frame, feature_columns = _build_weekly_ml_features(
        ml_source_df,
        extra_columns,
    )
    if len(feature_frame) < 10:
        raise ValueError(
            "At least 10 usable weekly rows are required after lag feature generation."
        )

    train_size = int(len(feature_frame) * 0.80)
    if train_size < 4 or len(feature_frame) - train_size < 1:
        raise ValueError("Not enough weekly rows for an 80/20 time-based split.")

    train = feature_frame.iloc[:train_size].copy()
    test = feature_frame.iloc[train_size:].copy()
    feature_columns, feature_exclusions = _select_model_features_from_train(
        train,
        feature_columns,
    )
    excluded_columns = [
        *excluded_columns,
        *feature_exclusions,
    ]
    if not feature_columns:
        raise ValueError("No usable forecast features remained after train-only filtering.")

    x_train = train[feature_columns]
    y_train = train["weekly_revenue"]
    x_test = test[feature_columns]
    y_test = test["weekly_revenue"]

    model_results = {
        "Moving Average Baseline": _evaluate_moving_average_baseline(weekly, test),
        "Ridge Regression": _fit_evaluate_ridge(
            sklearn,
            x_train,
            y_train,
            x_test,
            y_test,
        ),
        "Random Forest Regressor": _fit_evaluate_random_forest(
            sklearn,
            x_train,
            y_train,
            x_test,
            y_test,
        ),
    }

    selected_model = min(
        model_results,
        key=lambda name: (
            model_results[name]["metrics"]["sMAPE"],
            model_results[name]["metrics"]["RMSE"],
            model_results[name]["metrics"]["MAE"],
        ),
    )
    selected = model_results[selected_model]
    future_forecast = _forecast_future_weeks(
        selected_model,
        selected.get("model"),
        weekly,
        feature_frame,
        feature_columns,
        horizon,
    )
    chart_data = _build_forecast_chart_data(test, selected["predictions"], future_forecast)

    comparison_table = pd.DataFrame([
        {
            "model": name,
            "MAE": round(values["metrics"]["MAE"], 2),
            "RMSE": round(values["metrics"]["RMSE"], 2),
            "sMAPE": round(values["metrics"]["sMAPE"], 2),
            "Relative MAE": _round_optional(values["metrics"]["Relative MAE"], 2),
            "Relative RMSE": _round_optional(values["metrics"]["Relative RMSE"], 2),
        }
        for name, values in model_results.items()
    ]).sort_values(["sMAPE", "RMSE", "MAE"])

    return {
        "selected_model": selected_model,
        "compared_models": list(model_results.keys()),
        "selected_metrics": selected["metrics"],
        "forecasted_total_revenue": float(future_forecast["future_forecast"].sum()),
        "forecast_horizon": int(horizon),
        "chart_data": chart_data,
        "comparison_table": comparison_table,
        "feature_count": len(feature_columns),
        "extra_columns_used": extra_columns,
        "excluded_columns": excluded_columns,
        "future_forecast": future_forecast,
        "historical_weekly_revenue": weekly[["week_start", "weekly_revenue"]].copy(),
        "test_actual_average_revenue": _safe_mean(test["weekly_revenue"]),
    }


def _profile_extra_column(series, column):
    normalized_name = _normalize_name(column)
    tokens = set(normalized_name.split())
    non_missing = series.dropna()
    missing_ratio = float(series.isna().mean())
    numeric_values = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric_values.notna().mean())
    parsed_dates = pd.to_datetime(series, errors="coerce")
    date_ratio = float(parsed_dates.notna().mean())
    date_like_value_ratio = float(series.astype("string").map(_looks_like_date_text).mean())
    unique_count = int(non_missing.nunique())
    unique_ratio = float(unique_count / max(len(non_missing), 1))
    is_constant = unique_count <= 1
    has_identifier_keyword = _has_identifier_keyword(tokens, normalized_name)
    has_name_keyword = "name" in tokens

    return {
        "column": str(column),
        "normalized_name": normalized_name,
        "tokens": tokens,
        "missing_ratio": missing_ratio,
        "numeric_ratio": numeric_ratio,
        "date_parse_ratio": date_ratio,
        "date_like_value_ratio": date_like_value_ratio,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "is_constant": is_constant,
        "has_identifier_keyword": has_identifier_keyword,
        "has_name_keyword": has_name_keyword,
        "has_numeric_measurement_keyword": _has_numeric_measurement_keyword(tokens, normalized_name),
    }


def _infer_column_role(profile, selected_discount_column=None, column=None):
    if profile["is_constant"]:
        return "excluded", "constant column"
    if 1 - profile["missing_ratio"] < PROFILE_MIN_NON_MISSING_RATIO:
        return "excluded", "too many missing values"

    if selected_discount_column and column == selected_discount_column:
        return "numeric_measurement", "selected discount column"

    is_high_cardinality = (
        profile["unique_count"] > HIGH_CARDINALITY_UNIQUE_LIMIT
        or profile["unique_ratio"] > HIGH_CARDINALITY_RATIO_LIMIT
    )
    if profile["has_identifier_keyword"] and is_high_cardinality:
        return "identifier", "identifier keyword with high cardinality"
    if profile["has_identifier_keyword"] and profile["numeric_ratio"] >= 0.8:
        return "identifier", "numeric identifier-like column"

    if (
        profile["date_parse_ratio"] >= 0.8
        and profile["numeric_ratio"] < 0.5
        and _has_strong_date_signal(profile)
    ):
        return "secondary_date", "parseable secondary date"

    if profile["numeric_ratio"] >= 0.8:
        if profile["has_numeric_measurement_keyword"]:
            return "numeric_measurement", "numeric measurement profile"
        if profile["has_identifier_keyword"]:
            return "identifier", "numeric identifier-like column"
        if profile["unique_ratio"] <= HIGH_CARDINALITY_RATIO_LIMIT:
            return "numeric_measurement", "numeric profile with moderate cardinality"
        return "excluded", "numeric column does not look like a measurement"

    if is_high_cardinality:
        if profile["has_name_keyword"]:
            return "identifier", "name-like high-cardinality column"
        return "excluded", "high-cardinality categorical column"

    return "categorical_summary", "categorical summary profile"


def _build_weekly_ml_features(ml_source_df, extra_columns):
    prepared = ml_source_df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["quantity"] = pd.to_numeric(prepared["quantity"], errors="coerce")
    prepared["unit_price"] = pd.to_numeric(prepared["unit_price"], errors="coerce")
    prepared["revenue"] = pd.to_numeric(prepared["revenue"], errors="coerce")
    prepared = prepared.dropna(subset=["date"])
    prepared["week_start"] = prepared["date"].dt.to_period("W").apply(
        lambda period: period.start_time
    )

    weekly = prepared.groupby("week_start", dropna=True).agg(
        weekly_revenue=("revenue", "sum"),
        weekly_quantity=("quantity", "sum"),
        avg_unit_price=("unit_price", "mean"),
        product_count=("product_name", "nunique"),
        row_count=("date", "size"),
    )

    extra_feature_frames = []
    for column in extra_columns:
        if column not in prepared.columns:
            continue
        numeric_values = pd.to_numeric(prepared[column], errors="coerce")
        if numeric_values.notna().mean() >= 0.5:
            numeric_prepared = prepared.copy()
            numeric_prepared[column] = numeric_values
            numeric_weekly = numeric_prepared.groupby("week_start")[column].agg(
                ["mean", "sum", "min", "max"]
            )
            numeric_weekly.columns = [
                f"{_safe_feature_name(column)}_{stat}"
                for stat in numeric_weekly.columns
            ]
            extra_feature_frames.append(numeric_weekly)
        else:
            extra_feature_frames.append(_build_weekly_categorical_features(prepared, column))

    discount_columns = [
        column
        for column in extra_columns
        if "discount" in str(column).lower()
        or "disc" in str(column).lower()
        or "indirim" in str(column).lower()
    ]
    for column in discount_columns:
        if column in prepared.columns:
            extra_feature_frames.append(_build_weekly_discount_features(prepared, column))

    if extra_feature_frames:
        weekly = weekly.join(extra_feature_frames, how="left")

    weekly = weekly.sort_index().reset_index()
    calendar_features = ["year", "month", "quarter", "week_of_year"]
    weekly["year"] = weekly["week_start"].dt.year
    weekly["month"] = weekly["week_start"].dt.month
    weekly["quarter"] = weekly["week_start"].dt.quarter
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)

    lag_rolling_features = [
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_4",
        "rolling_revenue_4",
        "rolling_revenue_8",
        "quantity_lag_1",
        "rolling_quantity_4",
    ]
    weekly["revenue_lag_1"] = weekly["weekly_revenue"].shift(1)
    weekly["revenue_lag_2"] = weekly["weekly_revenue"].shift(2)
    weekly["revenue_lag_4"] = weekly["weekly_revenue"].shift(4)
    weekly["rolling_revenue_4"] = weekly["weekly_revenue"].shift(1).rolling(4).mean()
    weekly["rolling_revenue_8"] = weekly["weekly_revenue"].shift(1).rolling(8).mean()
    weekly["quantity_lag_1"] = weekly["weekly_quantity"].shift(1)
    weekly["rolling_quantity_4"] = weekly["weekly_quantity"].shift(1).rolling(4).mean()

    leakage_columns = {"weekly_revenue", "week_start"}
    allowed_direct_features = set(calendar_features + lag_rolling_features)
    current_week_summary_columns = [
        column
        for column in weekly.columns
        if column not in leakage_columns
        and column not in allowed_direct_features
    ]
    historical_summary_features = []
    for column in current_week_summary_columns:
        prev_column = f"prev_{column}"
        rolling_mean_column = f"rolling4_mean_{column}"
        rolling_sum_column = f"rolling4_sum_{column}"
        shifted = weekly[column].shift(1)
        weekly[prev_column] = shifted
        weekly[rolling_mean_column] = shifted.rolling(4).mean()
        weekly[rolling_sum_column] = shifted.rolling(4).sum()
        historical_summary_features.extend([
            prev_column,
            rolling_mean_column,
            rolling_sum_column,
        ])

    feature_columns = [
        column
        for column in calendar_features + lag_rolling_features + historical_summary_features
        if column in weekly.columns
    ]
    minimum_required_columns = [
        "weekly_revenue",
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_4",
        "rolling_revenue_4",
        "rolling_revenue_8",
        "quantity_lag_1",
        "rolling_quantity_4",
    ]
    minimum_required_columns = [
        column for column in minimum_required_columns if column in weekly.columns
    ]
    feature_frame = weekly.dropna(subset=minimum_required_columns).copy()
    feature_frame[feature_columns] = feature_frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return weekly, feature_frame, feature_columns


def _load_sklearn():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "RandomForestRegressor": RandomForestRegressor,
        "Ridge": Ridge,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def _select_model_features_from_train(train, feature_columns):
    train_week_count = len(train)
    max_features = max(1, min(60, train_week_count // 3))
    excluded = []
    scored_features = []
    priority_features = {
        "year",
        "month",
        "quarter",
        "week_of_year",
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_4",
        "rolling_revenue_4",
        "rolling_revenue_8",
        "quantity_lag_1",
        "rolling_quantity_4",
    }
    priority_selected = []

    for feature in feature_columns:
        values = pd.to_numeric(train[feature], errors="coerce")
        target = pd.to_numeric(train["weekly_revenue"], errors="coerce")
        missing_ratio = float(values.isna().mean())
        non_missing = values.dropna()
        unique_count = int(non_missing.nunique())
        variance = float(non_missing.var()) if len(non_missing) > 1 else 0

        if missing_ratio > 0.8:
            excluded.append({
                "column": feature,
                "role": "excluded_feature",
                "reason": "feature too sparse in train data",
            })
            continue
        if unique_count <= LOW_VARIANCE_UNIQUE_LIMIT or variance == 0:
            excluded.append({
                "column": feature,
                "role": "excluded_feature",
                "reason": "constant or low-variance train feature",
            })
            continue

        if feature in priority_features:
            priority_selected.append(feature)
            continue

        correlation = values.corr(target)
        if pd.isna(correlation):
            correlation = 0
        score = abs(float(correlation)) * (1 - missing_ratio)
        scored_features.append((feature, score))

    scored_features = sorted(scored_features, key=lambda item: item[1], reverse=True)
    remaining_slots = max(0, max_features - len(priority_selected))
    selected = [*priority_selected, *[feature for feature, _ in scored_features[:remaining_slots]]]
    for feature, _ in scored_features[remaining_slots:]:
        excluded.append({
            "column": feature,
            "role": "excluded_feature",
            "reason": f"feature limit exceeded (max_features={max_features})",
        })

    return selected, excluded


def _build_weekly_categorical_features(prepared, column):
    safe_column = _safe_feature_name(column)
    rows = []
    for week_start, group in prepared.groupby("week_start", dropna=True):
        values = group[column].dropna()
        top_share = 0
        if not values.empty:
            top_category = values.value_counts().index[0]
            top_revenue = group.loc[group[column] == top_category, "revenue"].sum()
            total_revenue = group["revenue"].sum()
            top_share = top_revenue / total_revenue if total_revenue else 0
        rows.append(
            {
                "week_start": week_start,
                f"{safe_column}_count": int(values.count()),
                f"{safe_column}_nunique": int(values.nunique()),
                f"{safe_column}_top_revenue_share": float(top_share),
            }
        )
    return pd.DataFrame(rows).set_index("week_start") if rows else pd.DataFrame()


def _build_weekly_discount_features(prepared, column):
    safe_column = _safe_feature_name(column)
    discount = pd.to_numeric(prepared[column], errors="coerce").map(_normalize_discount)
    discount_frame = prepared.assign(_discount=discount)
    grouped = discount_frame.groupby("week_start", dropna=True)
    features = grouped["_discount"].agg(["mean", "max"])
    features.columns = [
        f"{safe_column}_average_discount",
        f"{safe_column}_max_discount",
    ]
    features[f"{safe_column}_discounted_sales_ratio"] = grouped["_discount"].apply(
        lambda values: float((values.fillna(0) > 0).mean())
    )
    return features


def _evaluate_moving_average_baseline(weekly, test):
    predictions = []
    for _, row in test.iterrows():
        current_index = weekly.index[weekly["week_start"] == row["week_start"]][0]
        history = weekly.loc[: current_index - 1, "weekly_revenue"].tail(4)
        predictions.append(float(history.mean()) if not history.empty else 0)
    y_true = test["weekly_revenue"].to_numpy()
    y_pred = np.array(predictions)
    return {
        "model": None,
        "predictions": y_pred,
        "metrics": _forecast_metrics(y_true, y_pred),
    }


def _fit_evaluate_ridge(sklearn, x_train, y_train, x_test, y_test):
    model = sklearn["make_pipeline"](
        sklearn["SimpleImputer"](strategy="median"),
        sklearn["StandardScaler"](),
        sklearn["Ridge"](alpha=1.0),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "model": model,
        "predictions": predictions,
        "metrics": _forecast_metrics(y_test.to_numpy(), predictions),
    }


def _fit_evaluate_random_forest(sklearn, x_train, y_train, x_test, y_test):
    model = sklearn["make_pipeline"](
        sklearn["SimpleImputer"](strategy="median"),
        sklearn["RandomForestRegressor"](
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        ),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "model": model,
        "predictions": predictions,
        "metrics": _forecast_metrics(y_test.to_numpy(), predictions),
    }


def _forecast_future_weeks(model_name, model, weekly, feature_frame, feature_columns, horizon):
    revenue_history = weekly["weekly_revenue"].dropna().tolist()
    quantity_history = weekly["weekly_quantity"].dropna().tolist()
    last_week = pd.Timestamp(weekly["week_start"].max())
    template = feature_frame[feature_columns].iloc[-1].copy()
    forecasts = []

    for step in range(1, horizon + 1):
        week_start = last_week + pd.Timedelta(weeks=step)
        features = template.copy()
        features["year"] = week_start.year
        features["month"] = week_start.month
        features["quarter"] = week_start.quarter
        features["week_of_year"] = int(week_start.isocalendar().week)
        _set_if_present(features, "revenue_lag_1", _lag_value(revenue_history, 1))
        _set_if_present(features, "revenue_lag_2", _lag_value(revenue_history, 2))
        _set_if_present(features, "revenue_lag_4", _lag_value(revenue_history, 4))
        _set_if_present(features, "rolling_revenue_4", _tail_mean(revenue_history, 4))
        _set_if_present(features, "rolling_revenue_8", _tail_mean(revenue_history, 8))
        _set_if_present(features, "quantity_lag_1", _lag_value(quantity_history, 1))
        _set_if_present(features, "rolling_quantity_4", _tail_mean(quantity_history, 4))

        if model_name == "Moving Average Baseline":
            prediction = _tail_mean(revenue_history, 4)
        else:
            prediction = float(model.predict(pd.DataFrame([features], columns=feature_columns))[0])
        prediction = max(0, prediction)
        revenue_history.append(prediction)
        quantity_history.append(_tail_mean(quantity_history, 4))
        forecasts.append({"week_start": week_start, "future_forecast": prediction})

    future_forecast = pd.DataFrame(forecasts)
    future_forecast["future_forecast"] = future_forecast["future_forecast"].round(2)
    return future_forecast


def _build_forecast_chart_data(test, predictions, future_forecast):
    historical = pd.DataFrame(
        {
            "week_start": test["week_start"],
            "actual_revenue": test["weekly_revenue"],
            "predicted_revenue": predictions,
            "future_forecast": np.nan,
        }
    )
    future = future_forecast.copy()
    future["actual_revenue"] = np.nan
    future["predicted_revenue"] = np.nan
    return pd.concat(
        [
            historical,
            future[["week_start", "actual_revenue", "predicted_revenue", "future_forecast"]],
        ],
        ignore_index=True,
    )


def _forecast_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": float(np.mean(smape) * 100),
        "Relative MAE": _relative_error_percent(mae, y_true),
        "Relative RMSE": _relative_error_percent(rmse, y_true),
    }


def _relative_error_percent(error, y_true):
    actual_average = _safe_mean(y_true)
    if actual_average is None or actual_average == 0:
        return None
    return float(error / actual_average * 100)


def _safe_mean(values):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return None
    mean_value = float(numeric.mean())
    if not np.isfinite(mean_value):
        return None
    return mean_value


def _round_optional(value, digits):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _normalize_name(column):
    text = str(column).lower()
    normalized = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(normalized.split())


def _has_identifier_keyword(tokens, normalized_name):
    if tokens & IDENTIFIER_KEYWORDS:
        return True
    return any(
        token.endswith("id")
        or token.endswith("no")
        or token.endswith("code")
        for token in tokens
    ) or any(keyword in normalized_name for keyword in ["e mail", "phone number"])


def _has_numeric_measurement_keyword(tokens, normalized_name):
    if tokens & NUMERIC_MEASUREMENT_KEYWORDS:
        return True
    return any(keyword in normalized_name for keyword in NUMERIC_MEASUREMENT_KEYWORDS)


def _has_strong_date_signal(profile):
    tokens = profile["tokens"]
    if "date" in tokens or "time" in tokens:
        return True
    if tokens & {"delivery", "delivered"}:
        return True
    if "ship" in tokens and "mode" not in tokens:
        return True
    return profile["date_like_value_ratio"] >= 0.5


def _looks_like_date_text(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    has_separator = any(separator in text for separator in ["/", "-", "."])
    has_month_name = any(
        month in text.lower()
        for month in [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
    )
    return has_separator or has_month_name


def _date_delta_feature_name(column):
    normalized = _safe_feature_name(column).lower()
    if "ship" in normalized:
        return "ship_delay_days"
    if "delivery" in normalized or "delivered" in normalized:
        return "delivery_delay_days"
    return f"{normalized}_delay_days"


def _normalize_discount(value):
    if pd.isna(value):
        return np.nan
    if value > 1 and value <= 100:
        return value / 100
    return value


def _safe_feature_name(column):
    return "".join(ch if str(ch).isalnum() else "_" for ch in str(column)).strip("_")


def _safe_extra_column_name(column, existing_columns):
    column_name = str(column)
    if column_name not in existing_columns:
        return column_name

    base_name = f"extra_{column_name}"
    candidate = base_name
    suffix = 2
    while candidate in existing_columns:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def _set_if_present(series, key, value):
    if key in series.index:
        series[key] = value


def _lag_value(values, lag):
    if len(values) >= lag:
        return values[-lag]
    return values[-1] if values else 0


def _tail_mean(values, window):
    if not values:
        return 0
    return float(np.mean(values[-window:]))
