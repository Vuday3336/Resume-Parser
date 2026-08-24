"""Thin psycopg2 connection-pool wrapper. Raw SQL is used deliberately (not an ORM) so the
analytics queries and pgvector similarity search stay transparent and easy to reason about."""
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from psycopg2 import pool

from app.config import settings

_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        settings.require_database()
        _pool = pool.SimpleConnectionPool(1, 10, dsn=settings.DATABASE_URL)
    return _pool


@contextmanager
def get_connection():
    conn = _get_pool().getconn()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = True):
    with get_connection() as conn:
        cursor_factory = psycopg2.extras.RealDictCursor if dict_cursor else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur


def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def fetch_one(query: str, params: tuple = ()) -> dict | None:
    with get_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def execute(query: str, params: tuple = ()) -> None:
    with get_cursor() as cur:
        cur.execute(query, params)


def execute_returning_id(query: str, params: tuple = ()) -> int:
    with get_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()["id"]
