"""Markdown log writer for analysis output.

Turns analysis results (DataFrames, metric dicts, lists, code snippets) into
readable Markdown logs stored in the project's ``output/logs/`` directory. Tables
are generated directly, so it works with plain lists, dicts, and pandas
DataFrames alike.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def _format_cell(value: Any, float_format: Any, na_rep: str) -> str:
    """Render a single table cell as Markdown-safe text."""
    if value is None:
        return na_rep
    try:
        if pd.isna(value):
            return na_rep
    except (TypeError, ValueError):
        pass

    if isinstance(value, float):
        if float_format is not None:
            text = str(float_format(value))
        elif value.is_integer():
            text = str(int(value))
        else:
            text = str(value)
    else:
        text = str(value)

    return text.replace("|", r"\|").replace("\n", "<br>")


def _coerce_table(data: Any, headers: Sequence[Any] | None):
    """Normalise ``data`` into ``(rows, headers)``.

    Accepts a pandas DataFrame, a columnar dict, or an iterable of rows.
    """
    if isinstance(data, pd.DataFrame):
        if headers is None:
            headers = [str(column) for column in data.columns]
        rows = [tuple(row) for row in data.itertuples(index=False, name=None)]
        return rows, headers

    if isinstance(data, Mapping):
        keys = list(data.keys())
        if headers is None:
            headers = [str(key) for key in keys]
        columns = [list(data[key]) for key in keys]
        rows = [tuple(row) for row in zip(*columns)]
        return rows, headers

    rows = [tuple(row) for row in data]
    return rows, headers


class MarkdownLog:
    """Write Markdown sections to a named log file.

    A ``.md`` extension is added when missing and ``output_dir`` is created if
    necessary. Use as a context manager (``with MarkdownLog(...) as log:``) or
    close manually via :meth:`close`; a best-effort :meth:`__del__` hook also
    closes the file if neither is done. ``append=True`` keeps existing content.
    """

    def __init__(
        self,
        name: str | Path,
        output_dir: str | Path = LOGS_DIR,
        append: bool = False,
    ) -> None:
        self._file = None  # set first so __del__ is safe even if init fails

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        filename = str(name)
        if not filename.lower().endswith(".md"):
            filename += ".md"
        self.path = self.output_dir / filename

        self._file = open(self.path, "a" if append else "w", encoding="utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        """Best-effort close when the caller forgets ``close()`` / ``with``.

        The context manager is the deterministic guarantee; this is only a
        safety net for the non-context usage pattern.
        """
        try:
            if self._file is not None and not self._file.closed:
                self._file.close()
        except Exception:
            pass

    def write(self, text: str = "") -> None:
        """Write raw text followed by a newline."""
        self._file.write(text.rstrip("\n") + "\n")

    def blank(self) -> None:
        self.write("")

    def rule(self) -> None:
        self.write("---")
        self.blank()

    def heading(self, text: str, level: int = 1) -> None:
        self.write(f"{'#' * level} {text}")

    def paragraph(self, text: str) -> None:
        self.write(text)
        self.blank()

    def bullets(self, items: Iterable[Any]) -> None:
        for item in items:
            self.write(f"- {item}")
        self.blank()

    def numbered(self, items: Iterable[Any]) -> None:
        for index, item in enumerate(items, start=1):
            self.write(f"{index}. {item}")
        self.blank()

    def code(self, text: str, language: str = "") -> None:
        self.write(f"```{language}")
        self.write(text)
        self.write("```")
        self.blank()

    def table(
        self,
        data: Any,
        headers: Sequence[Any] | None = None,
        *,
        float_format: Any = None,
        na_rep: str = "",
    ) -> None:
        """Render ``data`` as a Markdown table.

        Missing values render as ``na_rep``; floats can be formatted via
        ``float_format`` (e.g. ``"{:.2f}".format``).
        """
        rows, headers = _coerce_table(data, headers)

        width = len(headers) if headers else 0
        for row in rows:
            width = max(width, len(row))
        if width == 0:
            return

        header_cells: list[Any] = list(headers) if headers else [""] * width
        header_cells += [""] * (width - len(header_cells))

        def fmt(value: Any) -> str:
            return _format_cell(value, float_format, na_rep)

        lines = [
            "| " + " | ".join(fmt(cell) for cell in header_cells) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        for row in rows:
            cells = list(row) + [""] * (width - len(row))
            lines.append(
                "| " + " | ".join(fmt(cell) for cell in cells[:width]) + " |"
            )

        self.write("\n".join(lines))
        self.blank()

    def key_values(
        self,
        pairs: Mapping[Any, Any] | Sequence[Sequence[Any]],
        *,
        float_format: Any = None,
        na_rep: str = "",
    ) -> None:
        """Write a two-column ``Key | Value`` table."""
        items = list(pairs.items()) if isinstance(pairs, Mapping) else list(pairs)
        rows = [list(pair) for pair in items]
        self.table(rows, headers=["Key", "Value"], float_format=float_format, na_rep=na_rep)

    def header(self, title: str, level: int = 1) -> None:
        """Write a titled section header plus a generation timestamp."""
        self.heading(title, level)
        self.write(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        self.blank()

    def close(self) -> None:
        """Close the underlying file (safe to call more than once)."""
        if self._file is not None and not self._file.closed:
            self._file.close()


def write_table(
    data: Any,
    name: str | Path,
    title: str | None = None,
    *,
    output_dir: str | Path = LOGS_DIR,
    append: bool = False,
    float_format: Any = None,
    na_rep: str = "",
) -> Path:
    """Write a single table to a named log and return its path."""
    with MarkdownLog(name, output_dir=output_dir, append=append) as log:
        if title:
            log.heading(title, level=2)
        log.table(data, float_format=float_format, na_rep=na_rep)
    return log.path


__all__ = [
    "FIGURES_DIR",
    "LOGS_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "MarkdownLog",
    "write_table",
]


if __name__ == "__main__":
    log = MarkdownLog("_demo_log")
    with log:
        log.header("Demo Log")
        log.paragraph("This file demonstrates the MarkdownLog API.")
        log.heading("Lists", level=2)
        log.bullets(["First item", "Second item"])
        log.numbered(["Step one", "Step two"])
        log.code("print('hello')", language="python")
        log.key_values({"n_rows": 750, "n_columns": 18})
        log.heading("Table", level=2)
        log.table(
            [["C0001", 2025, 4.8], ["C0002", 2023, 3.7]],
            headers=["Concept_ID", "Test_Year", "Appeal"],
        )
    print(f"Demo written to: {log.path}")

