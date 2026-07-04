import json
import os
from pathlib import Path

from google import genai


LLM_NOT_RUNNING_MESSAGE = (
    "Gemini API is not available. Check internet connection or API quota."
)


def _load_local_env_file():
    """Load simple KEY=VALUE or export KEY=VALUE lines from project .env file."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        # Do not overwrite values already exported in the terminal.
        os.environ.setdefault(key, value)


_load_local_env_file()


def generate_ai_insights(summary):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env or export it before running Streamlit."
        )

    if not model:
        model = "gemini-3.1-flash-lite"

    client = genai.Client(api_key=api_key)

    system_instruction = (
        "You explain sales analytics to a small business owner. "
        "Use plain business language. Avoid technical terms. "
        "Keep the answer short, clear, and action-oriented. "
        "Return standard Markdown. Use **bold** for the most important figures, "
        "products, risks, and actions when it improves clarity. "
        "Use bullet points when listing recommended actions. "
        "Do not use inline code, code blocks, backticks, HTML, tables, or color syntax. "
        "Use exactly these section headings: "
        "General Summary, Revenue Drivers, Risks and Anomalies, "
        "Forecast Interpretation, Recommended Actions. "
        "In Forecast Interpretation, read the multi-horizon forecast summary "
        "for 1, 2, 4, 8, and 12 weeks together. Compare the short-term view, "
        "the medium-term view, and the full 12-week outlook. Explain the most "
        "important differences, trend direction, previous-period comparison, "
        "and conservative/expected/optimistic range in friendly business "
        "language without repeating every number one by one."
    )

    user_prompt = (
        "Create concise business insights from this JSON summary. "
        "Do not mention that raw data was not provided. "
        "Do not invent numbers, products, risks, or forecasts that are not present in the JSON. "
        "If a section has missing data, say it is not available in one short sentence. "
        "For the forecast, use the model-level metrics once and interpret the "
        "1, 2, 4, 8, and 12-week summaries as one connected forecast story.\n\n"
        f"{json.dumps(summary, ensure_ascii=False, indent=2)}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.2,
                "max_output_tokens": 900,
            },
        )
    except Exception as exc:
        raise RuntimeError(LLM_NOT_RUNNING_MESSAGE) from exc

    content = getattr(response, "text", None)
    if not content or not content.strip():
        raise RuntimeError(LLM_NOT_RUNNING_MESSAGE)

    return content.strip()
