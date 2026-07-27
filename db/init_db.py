"""One-shot script: create tables in the local alpine-postgres instance.

Usage:
    docker compose up -d db
    python -m db.init_db
"""
from db.models import init_db

if __name__ == "__main__":
    init_db()
    print("spacethink: tables created in Postgres (alpine).")
