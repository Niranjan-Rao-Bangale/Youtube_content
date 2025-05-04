# server_sqlite.py  – works with FastMCP ≥ 1.0  (no “servers” pkg)
from pathlib import Path
from fastmcp import FastMCP
import sqlite3

mcp = FastMCP("SQLite Explorer")

DB_PATH = Path(r"<your sqlite.db>")

def _con():
    return sqlite3.connect(DB_PATH)

@mcp.tool()
def list_tables() -> list[str]:
    """Return all table names."""
    with _con() as c:
        rows = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return [r[0] for r in rows]

@mcp.tool()
def describe_table(table: str) -> list[dict]:
    """Return column metadata for a table."""
    with _con() as c:
        rows = c.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [
        {
            "cid": cid,
            "name": name,
            "type": col_type,
            "notnull": bool(notnull),
            "default": dflt_value,
            "pk": bool(pk),
        }
        for cid, name, col_type, notnull, dflt_value, pk in rows
    ]

@mcp.tool()
def read_query(sql: str) -> list[dict]:
    """Run a **validated SELECT** and return up to 2 000 rows."""
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    with _con() as c:
        c.row_factory = sqlite3.Row
        print("sql:", sql)
        rows = c.execute(sql).fetchmany(2000)
    return [dict(r) for r in rows]


if __name__ == "__main__":
    mcp.run(transport="stdio")        # lets Agents spawn it
