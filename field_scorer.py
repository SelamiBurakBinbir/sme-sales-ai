from column_profiler import parse_numeric


TARGET_FIELDS = ["date", "product_name", "quantity", "unit_price", "revenue"]
DISCOUNT_KEYWORDS = ["discount", "disc", "indirim"]
BASE_REVENUE_FORMULA = "quantity * unit_price"
DISCOUNTED_REVENUE_FORMULA = "quantity * unit_price * (1 - discount)"


def score_basic_fields(profiles):
    return {
        "date": _score_date(profiles),
        "product_name": _score_product_name(profiles),
        "quantity": _score_quantity(profiles),
        "unit_price": _score_unit_price(profiles),
    }


def score_revenue(
    profiles,
    df,
    selected_quantity=None,
    selected_unit_price=None,
    selected_discount=None,
):
    candidates = []

    for column, profile in profiles.items():
        normalized = profile["normalized_name"]
        dominant_type = profile["dominant_type"]

        a = _contains_any(normalized, ["revenue"])
        b = _contains_any(normalized, ["total", "toplam"])
        c = _contains_any(normalized, ["amount"])
        d = 0 if _contains_any(
            normalized,
            ["unit", "birim", "quantity", "qty", "adet", "miktar"],
        ) else 1
        e = 1 if dominant_type == "numeric" else 0
        f = 1 if profile["has_currency"] else 0
        z = int(validate_revenue_formula(
            df,
            column,
            selected_quantity,
            selected_unit_price,
            selected_discount,
        )["is_valid"])

        score = d * e * min(1, max(a, z, b * 0.8 + c * 0.2 + f * 0.2))
        if score >= 0.8:
            candidates.append(_candidate(column, score))

    return _sort_candidates(candidates)


def score_discount(profiles):
    candidates = []
    for column, profile in profiles.items():
        normalized = profile["normalized_name"]
        dominant_type = profile["dominant_type"]

        a = _contains_any(normalized, DISCOUNT_KEYWORDS)
        b = 1 if dominant_type in {"numeric", "short_string"} else 0
        c = 0 if _contains_any(
            normalized,
            ["price", "fiyat", "revenue", "total", "toplam", "quantity", "qty", "adet"],
        ) else 1

        score = a * b * c
        if score >= 0.5:
            candidates.append(_candidate(column, score))

    return _sort_candidates(candidates)


def _score_date(profiles):
    candidates = []
    for column, profile in profiles.items():
        normalized = profile["normalized_name"]
        a = _contains_any(normalized, ["date", "tarih"])
        b = 1 if profile["dominant_type"] == "date" else 0
        score = a * 0.5 + b * 0.5
        if score >= 0.5:
            candidates.append(_candidate(column, score))
    return _sort_candidates(candidates)


def _score_product_name(profiles):
    candidates = []
    for column, profile in profiles.items():
        normalized = profile["normalized_name"]
        a = _contains_any(normalized, ["product", "item", "urun", "description"])
        b = _contains_any(normalized, ["name", "isim", "ismi", "detail"])
        c = 0 if profile["dominant_type"] in {"date", "numeric"} else 1
        score = c * (a * 0.7 + b * 0.3)
        if score >= 0.3:
            candidates.append(_candidate(column, score))
    return _sort_candidates(candidates)


def _score_quantity(profiles):
    candidates = []
    for column, profile in profiles.items():
        normalized = profile["normalized_name"]
        dominant_type = profile["dominant_type"]

        a = _contains_any(normalized, ["quantity", "qty", "adet", "miktar", "units"])
        b = _contains_any(normalized, ["amount"])
        c = 0 if _contains_any(
            normalized,
            ["price", "fiyat", "spent", "cost", "revenue", "total", "toplam"],
        ) else 1
        d = 0 if dominant_type in {"date", "long_string"} else 1

        if dominant_type == "numeric":
            e = 1
        elif dominant_type == "short_string":
            e = 0.5
        else:
            e = 0

        score = c * d * (min(1, a + b * 0.5) * 0.6 + e * 0.4)
        if score >= 0.5:
            candidates.append(_candidate(column, score))
    return _sort_candidates(candidates)


def _score_unit_price(profiles):
    candidates = []
    for column, profile in profiles.items():
        normalized = profile["normalized_name"]

        a = _contains_any(normalized, ["price", "fiyat"])
        b = _contains_any(normalized, ["unit", "birim"])
        c = 0 if _contains_any(
            normalized,
            ["total", "toplam", "quantity", "qty", "adet", "miktar"],
        ) else 1
        d = 1 if profile["dominant_type"] == "numeric" else 0
        e = 1 if profile["has_currency"] else 0

        score = c * d * min(1, a * 0.6 + b * 0.4 + e * 0.2)
        if score >= 0.6:
            candidates.append(_candidate(column, score))
    return _sort_candidates(candidates)


def validate_revenue_formula(
    df,
    revenue_column,
    quantity_column,
    unit_price_column,
    discount_column=None,
):
    result = {
        "is_valid": False,
        "checked_rows": 0,
        "matched_rows": 0,
        "match_ratio": 0,
        "formula": BASE_REVENUE_FORMULA,
        "formula_label": "quantity × unit_price",
        "formula_results": [],
        "details": [],
    }

    if not quantity_column or not unit_price_column:
        return result
    if revenue_column in {quantity_column, unit_price_column, discount_column}:
        return result
    if discount_column in {quantity_column, unit_price_column, revenue_column}:
        discount_column = None

    formulas = [_build_revenue_formula(BASE_REVENUE_FORMULA, None)]
    if discount_column:
        formulas.append(_build_revenue_formula(DISCOUNTED_REVENUE_FORMULA, discount_column))

    required_columns = [quantity_column, unit_price_column, revenue_column]
    if discount_column:
        required_columns.append(discount_column)

    sample = df[required_columns].head(10)
    formula_results = [_evaluate_revenue_formula(
        sample,
        revenue_column,
        quantity_column,
        unit_price_column,
        formula,
    ) for formula in formulas]

    best_result = max(
        formula_results,
        key=lambda item: (item["match_ratio"], item["matched_rows"], item["checked_rows"]),
    )

    result.update({
        "is_valid": best_result["is_valid"],
        "checked_rows": best_result["checked_rows"],
        "matched_rows": best_result["matched_rows"],
        "match_ratio": best_result["match_ratio"],
        "formula": best_result["formula"],
        "formula_label": best_result["formula_label"],
        "formula_results": [
            {
                "formula": item["formula"],
                "formula_label": item["formula_label"],
                "checked_rows": item["checked_rows"],
                "matched_rows": item["matched_rows"],
                "match_ratio": item["match_ratio"],
                "is_valid": item["is_valid"],
            }
            for item in formula_results
        ],
        "details": best_result["details"],
    })

    return result


def _build_revenue_formula(formula, discount_column):
    return {
        "formula": formula,
        "formula_label": formula.replace("*", "×"),
        "discount_column": discount_column,
    }


def _evaluate_revenue_formula(
    sample,
    revenue_column,
    quantity_column,
    unit_price_column,
    formula,
):
    result = {
        "is_valid": False,
        "checked_rows": 0,
        "matched_rows": 0,
        "match_ratio": 0,
        "formula": formula["formula"],
        "formula_label": formula["formula_label"],
        "details": [],
    }

    for row_index, row in sample.iterrows():
        quantity = parse_numeric(row[quantity_column])
        unit_price = parse_numeric(row[unit_price_column])
        revenue = parse_numeric(row[revenue_column])
        discount = _parse_discount(row[formula["discount_column"]]) if formula["discount_column"] else None

        if quantity is None or unit_price is None or revenue is None or revenue == 0:
            continue
        if formula["discount_column"] and discount is None:
            continue

        predicted_revenue = quantity * unit_price
        if formula["discount_column"]:
            predicted_revenue *= 1 - discount
        result["checked_rows"] += 1
        relative_error = abs(predicted_revenue - revenue) / abs(revenue)
        is_match = relative_error <= 0.05

        if is_match:
            result["matched_rows"] += 1

        result["details"].append({
            "row_index": int(row_index),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "predicted_revenue": predicted_revenue,
            "relative_error": round(relative_error, 6),
            "is_match": is_match,
        })

    if result["checked_rows"] > 0:
        result["match_ratio"] = result["matched_rows"] / result["checked_rows"]
        result["is_valid"] = result["match_ratio"] > 0.5

    return result


def _parse_discount(value):
    discount = parse_numeric(value)
    if discount is None:
        return None
    if discount > 1 and discount <= 100:
        return discount / 100
    return discount


def _contains_any(text, fragments):
    return int(any(fragment in text for fragment in fragments))


def _candidate(column, score):
    return {"column": column, "score": round(score, 4)}


def _sort_candidates(candidates):
    return sorted(candidates, key=lambda item: item["score"], reverse=True)
