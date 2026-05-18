from __future__ import annotations

import argparse
import base64
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


DEFAULT_DB = "gateway.db"
DEFAULT_OUTPUT = "gateway_export.json"

JSON_COLUMNS = {
    ("job_environments", "env_params"),
    ("session_steps", "messages"),
    ("session_steps", "env_state"),
}

BOOLEAN_COLUMNS = {
    ("job_environments", "finished"),
    ("job_environments", "is_deleted"),
    ("session_steps", "is_terminal"),
    ("session_steps", "is_truncated"),
    ("session_steps", "is_session_completed"),
    ("session_steps", "is_trainable"),
}


def export_gateway_db(
    db: str | Path = DEFAULT_DB,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    job_id: str | None = None,
    tables: list[str] | None = None,
    parse_json_fields: bool = True,
    indent: int | None = 2,
) -> Path:
    db_path = _resolve_sqlite_path(db)
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    with _connect_readonly(db_path) as conn:
        conn.row_factory = sqlite3.Row
        available_tables = _list_tables(conn)
        selected_tables = _select_tables(available_tables, tables)

        exported_tables: dict[str, list[dict[str, Any]]] = {}
        row_counts: dict[str, int] = {}
        for table in selected_tables:
            columns = _table_columns(conn, table)
            rows = _fetch_rows(conn, table, columns, job_id=job_id)
            exported_tables[table] = [
                _normalize_row(table, row, parse_json_fields=parse_json_fields)
                for row in rows
            ]
            row_counts[table] = len(exported_tables[table])

    payload = {
        "metadata": {
            "source_db": str(db_path),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "tables": selected_tables,
            "row_counts": row_counts,
        },
        "tables": exported_tables,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=indent)
        fh.write("\n")

    return output_path


def _resolve_sqlite_path(db: str | Path) -> Path:
    value = str(db)
    if value.startswith("sqlite://"):
        value = value[len("sqlite://") :].split("?", 1)[0]

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    if not path.is_file():
        raise ValueError(f"SQLite database path is not a file: {path}")

    return path


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri_path = quote(path.as_posix())
    return sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [str(row[0]) for row in cursor.fetchall()]


def _select_tables(available_tables: list[str], requested_tables: list[str] | None) -> list[str]:
    if not requested_tables:
        return available_tables

    available = set(available_tables)
    missing = sorted(set(requested_tables) - available)
    if missing:
        raise ValueError(
            "Requested table(s) not found: "
            + ", ".join(missing)
            + ". Available tables: "
            + ", ".join(available_tables)
        )

    return [table for table in available_tables if table in set(requested_tables)]


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cursor = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    return [str(row[1]) for row in cursor.fetchall()]


def _fetch_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    *,
    job_id: str | None,
) -> list[sqlite3.Row]:
    where_clause = ""
    params: list[Any] = []
    if job_id is not None and "job_id" in columns:
        where_clause = " WHERE job_id = ?"
        params.append(job_id)

    order_clause = " ORDER BY id ASC" if "id" in columns else ""
    cursor = conn.execute(
        f"SELECT * FROM {_quote_identifier(table)}{where_clause}{order_clause}",
        params,
    )
    return cursor.fetchall()


def _normalize_row(
    table: str,
    row: sqlite3.Row,
    *,
    parse_json_fields: bool,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for column in row.keys():
        value = row[column]
        if (table, column) in BOOLEAN_COLUMNS and value in (0, 1):
            normalized[column] = bool(value)
        elif parse_json_fields and (table, column) in JSON_COLUMNS:
            normalized[column] = _parse_json_value(value)
        else:
            normalized[column] = _json_safe_value(value)
    return normalized


def _parse_json_value(value: Any) -> Any:
    if value in (None, ""):
        return value
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return _json_safe_value(value)

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    return value


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the Safactory gateway SQLite database to a JSON file.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=(
            "Gateway SQLite database path or sqlite:// URL. "
            f"Defaults to {DEFAULT_DB!r}."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination JSON file. Defaults to {DEFAULT_OUTPUT!r}.",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Only export rows for this job_id when a table has a job_id column.",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Export only this table. Can be provided multiple times.",
    )
    parser.add_argument(
        "--raw-json-fields",
        action="store_true",
        help="Keep JSON-valued database columns as raw strings instead of parsing them.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty-printed JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = export_gateway_db(
            db=args.db,
            output=args.output,
            job_id=args.job_id,
            tables=args.tables,
            parse_json_fields=not args.raw_json_fields,
            indent=None if args.compact else 2,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))

    print(f"Exported gateway database to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
