from __future__ import annotations

from pathlib import Path
from typing import List

from .. import config
from ..core import db


def split_statements(raw: str) -> List[str]:


    stmts: List[str] = []
    buf: List[str] = []
    for line in raw.splitlines():
        st = line.strip()
        if st == "/":
            s = "\n".join(buf).strip()
            if s:
                stmts.append(s)
            buf = []
            continue
        buf.append(line)
        joined = "\n".join(buf)
        if "BEGIN" not in joined.upper() and st.endswith(";"):
            s = joined.strip().rstrip(";").strip()
            if s:
                stmts.append(s)
            buf = []
    tail = "\n".join(buf).strip().rstrip(";").strip()
    if tail:
        stmts.append(tail)
    return stmts


def run_sql_file(cur, path: Path) -> None:
    print(f"  Ejecutando {path.name} ...")
    for stmt in split_statements(path.read_text(encoding="utf-8")):
        cur.execute(stmt)


def main() -> None:
    print("== Bootstrap: esquema ==")
    con = db.connect(retries=3, delay=4.0)
    try:
        cur = con.cursor()
        run_sql_file(cur, config.SQL_DIR / "01_schema.sql")
        con.commit()

        print("  OK. Esquema creado (tablas, tipo VECTOR, intento de indice vectorial).")
    finally:
        con.close()


if __name__ == "__main__":
    main()
