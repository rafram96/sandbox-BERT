"""Conexion a Oracle 23ai (python-oracledb en modo thin: sin Instant Client)."""
import array
import time
import oracledb

from . import config


def connect(retries: int = 1, delay: float = 3.0) -> oracledb.Connection:
    """Abre una conexion. Con retries>1 reintenta (util mientras la BD arranca)."""
    last = None
    for intento in range(1, retries + 1):
        try:
            return oracledb.connect(
                user=config.ORA_USER,
                password=config.ORA_PASSWORD,
                dsn=config.ORA_DSN,
            )
        except oracledb.Error as e:
            last = e
            if intento < retries:
                print(f"  BD no lista (intento {intento}/{retries}), reintentando en {delay:.0f}s...")
                time.sleep(delay)
    raise last


def wait_until_ready(timeout: float = 180.0) -> None:
    """Bloquea hasta que la BD acepte conexiones o se agote el timeout."""
    deadline = time.time() + timeout
    while True:
        try:
            con = connect(retries=1)
            con.close()
            return
        except oracledb.Error as e:
            if time.time() > deadline:
                raise TimeoutError(f"La BD no estuvo lista en {timeout:.0f}s: {e}")
            time.sleep(3.0)


def to_vector(values) -> array.array:
    """Convierte una lista/np.array de floats al formato que espera el tipo VECTOR."""
    return array.array("f", (float(x) for x in values))
