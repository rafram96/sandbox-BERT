"""Muestra las clasificaciones persistidas en Oracle.  Uso: python -m src.scripts.query_resultados"""
from ..core import db


def main() -> None:
    con = db.connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT d.expediente, cl.categoria_base, cl.score_confianza,
                   cl.ruta, cl.escalo_llm, cl.etiqueta_final, cl.revision_humana
            FROM clasificaciones cl
            JOIN documentos d ON d.id = cl.documento_id
            ORDER BY cl.id
            """
        )
        print(f"{'EXPEDIENTE':<16}{'BASE':<22}{'CONF':>6}  {'RUTA':<8}{'ESCALO':<8}{'FINAL':<22}{'REV'}")
        print("-" * 92)
        for exp, base, conf, ruta, escalo, final, rev in cur:
            print(f"{(exp or '-'):<16}{base:<22}{conf:>6.2f}  {ruta:<8}"
                  f"{('Si' if escalo else 'No'):<8}{final:<22}{'Si' if rev else 'No'}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
