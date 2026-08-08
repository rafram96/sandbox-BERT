import array
import time
import oracledb

from .. import config


def connect(retries: int = 1, delay: float = 3.0) -> oracledb.Connection:

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

    return array.array("f", (float(x) for x in values))
