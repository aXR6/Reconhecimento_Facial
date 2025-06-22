import logging
import os

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2 import extras
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psycopg2 = None
    extras = None

logger = logging.getLogger(__name__)

DSN = os.getenv("POSTGRES_DSN")

if not DSN:
    logger.warning("POSTGRES_DSN not set; database features disabled")


@contextmanager
def get_conn():
    if psycopg2 is None or not DSN:
        if psycopg2 is None:
            logger.error("psycopg2 not installed")
        else:
            logger.error("POSTGRES_DSN not configured")
        yield None
        return
    conn = None
    try:
        conn = psycopg2.connect(DSN)
        yield conn
    except Exception as exc:  # noqa: BLE001
        logger.error("DB connection error: %s", exc)
        yield None
    finally:
        if conn:
            conn.close()


def init_db() -> None:
    """Create tables if they do not exist."""
    with get_conn() as conn:
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id SERIAL PRIMARY KEY,
                image TEXT,
                faces INTEGER,
                caption TEXT,
                obstruction TEXT,
                recognized TEXT,
                result_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                name TEXT,
                embedding BYTEA
            );
            """
        )
        conn.commit()


def save_detection(image: str, faces: int, caption: str = "", obstruction: str = "", recognized: str = "", result_json: dict | None = None) -> None:
    with get_conn() as conn:
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO detections (image, faces, caption, obstruction, recognized, result_json) VALUES (%s, %s, %s, %s, %s, %s)",
            (image, faces, caption, obstruction, recognized, extras.Json(result_json) if extras else None),
        )
        conn.commit()
