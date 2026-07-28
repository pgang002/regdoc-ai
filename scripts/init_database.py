from __future__ import annotations

from pathlib import Path

from regdoc_ai.persistence import Database


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    database = Database.from_env(PROJECT_ROOT)
    database.create_schema()
    if not database.healthcheck():
        raise SystemExit("Database health check failed")
    print(f"Database schema ready: {database.url}")


if __name__ == "__main__":
    main()
