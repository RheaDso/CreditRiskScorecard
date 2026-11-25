"""M13: render the model report and run summary from computed results only."""

from pathlib import Path
from typing import Any

import markdown
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent / "templates"


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def write_model_report(context: dict[str, Any], reports_dir: Path) -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATES), keep_trailing_newline=True)
    values = {
        k: md_table(v) if isinstance(v, pd.DataFrame) else v for k, v in context.items()
    }
    text = env.get_template("model_report.md.j2").render(**values)
    (reports_dir / "model_report.md").write_text(text, encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables"])
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Model report</title><style>"
        "body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
        "img{max-width:100%}</style></head><body>" + body + "</body></html>"
    )
    (reports_dir / "model_report.html").write_text(html, encoding="utf-8")


def write_run_summary(
    rows: list[tuple[str, str, str, bool]], notes: list[str], reports_dir: Path, synthetic: bool
) -> None:
    table = pd.DataFrame(
        [(m, t, a, "yes" if ok else "NO") for m, t, a, ok in rows],
        columns=["metric", "target", "actual", "met"],
    )
    data = "synthetic data" if synthetic else "Home Credit data"
    text = (
        f"# Run summary ({data})\n\n{md_table(table)}\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in notes)
        + "\n\nTest count and coverage: run `pytest`. "
        "Deviations: see [DECISIONS.md](../DECISIONS.md).\n"
    )
    (reports_dir / "RUN_SUMMARY.md").write_text(text, encoding="utf-8")
