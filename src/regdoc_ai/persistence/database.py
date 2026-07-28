from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DEFAULT_SQLITE_PATH = Path("runtime/day9/regdoc_ai.db")


def database_url_from_env(project_root: str | Path | None = None) -> str:
    configured = os.getenv("REGDOC_DATABASE_URL")
    if configured:
        return configured
    root = Path(project_root or Path.cwd()).resolve()
    db_path = (root / DEFAULT_SQLITE_PATH).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


class Database:
    """SQLAlchemy 2.x database wrapper.

    PostgreSQL is the production target. SQLite is retained as a deterministic local
    test mode so the metadata and job-state logic can be executed without external
    services.
    """

    def __init__(self, url: str, *, echo: bool = False):
        kwargs: dict[str, object] = {"pool_pre_ping": True, "echo": echo}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.url = url
        self.engine: Engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _configure_sqlite(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        self.SessionLocal = sessionmaker(
            bind=self.engine, class_=Session, expire_on_commit=False, autoflush=False
        )

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> Database:
        return cls(database_url_from_env(project_root))

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self.engine)

    def session(self) -> Session:
        return self.SessionLocal()

    def healthcheck(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
