"""Ejecuta los .sql de /sql para crear el esquema. python -m src.bootstrap"""
from __future__ import annotations

from pathlib import Path
from typing import List

from . import config, db


def split_statements(raw: str) -> List[str]:
    """Separa un script estilo SQL*Plus en sentencias ejecutables por oracledb.

    Reglas (suficientes para los scripts controlados de este sandbox):
      - Una linea que contiene solo '/' termina un bloque PL/SQL.
      - Fuera de un bloque (sin BEGIN acumulado), ';' termina la sentencia.
    """
    stmts: List[str] = []
    buf: List[str] = []
    for line in raw.splitlines():
        st = line.strip()
        if st == "/":
            s = "\n".join(buf).strip()
            if s:
                stmts.append(s)          # bloque PL/SQL: se ejecuta tal cual (con END;)
            buf = []
            continue
        buf.append(line)
        joined = "\n".join(buf)
        if "BEGIN" not in joined.upper() and st.endswith(";"):
            s = joined.strip().rstrip(";").strip()
            if s:
                stmts.append(s)          # sentencia plana: sin ';' final
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
        # Las categorias se derivan del corpus en src/ingest.py (dataset-agnostico).
        # 02_seed.sql queda como referencia ilustrativa de SUNAFIL, no se ejecuta aqui.
        print("  OK. Esquema creado (tablas, tipo VECTOR, intento de indice vectorial).")
    finally:
        con.close()


if __name__ == "__main__":
    main()
