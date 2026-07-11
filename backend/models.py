# backend/models.py (FIXED)
"""
Two real bugs fixed here:

1. The entire original file was indented by 4 spaces starting from the
   very first line (`    import os`), with no enclosing block. That's not
   legal Python at module level — `import models` raised
   `IndentationError: unexpected indent` immediately, before any of the
   DB logic even ran. This matters now more than before: main.py's new
   /internal/qstash-webhook handler does `from models import SessionLocal,
   StageEvent`, so this file being unimportable would break event logging
   the moment QStash is actually wired up (which you've now confirmed you
   want working for real).

2. `db.execute("SELECT 1")` — SQLAlchemy 1.4+/2.0 requires raw SQL strings
   to be wrapped in `text(...)`, otherwise it raises
   `ObjectNotExecutableError: Not an executable object: 'SELECT 1'`.

Also note: this file (and daily_analytics_runner.py) need `sqlalchemy` AND
a Postgres driver (`psycopg2-binary`) installed — neither was in
requirements.txt. Added both.
"""

import os
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text, pool, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Database URL Configuration ─────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://polyglot:polyglot_pass@localhost:5432/polyglot_echo"
)

print(f"[DB Config] Using DATABASE_URL: {DATABASE_URL[:50]}...")

# ── Engine Configuration with Connection Pooling ────────────────
# NullPool is right for this — serverless/single-container deployments
# shouldn't hold a persistent connection pool open.
try:
    engine = create_engine(
        DATABASE_URL,
        poolclass=pool.NullPool,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 10,
            "application_name": "polyglot_echo"
        },
        echo=False
    )
    print("[DB Config] ✅ Engine created successfully")
except Exception as e:
    print(f"[DB Config] ❌ Failed to create engine: {e}")
    raise

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ── Table Definitions ──────────────────────────────────────────
class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(String, index=True)
    input_text      = Column(Text)
    input_lang      = Column(String(10))
    output_text     = Column(Text)
    output_lang     = Column(String(10))
    speaker_profile = Column(String(20))
    whisper_ms      = Column(Integer)
    llm_ms          = Column(Integer)
    tts_ms          = Column(Integer)
    total_ms        = Column(Integer)
    cache_hit       = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow, index=True)


class StageEvent(Base):
    __tablename__ = "stage_events"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String, index=True)
    event_type    = Column(String(50))
    latency_ms    = Column(Integer)
    metadata_json = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)


# ── Table Creation with Error Handling ──────────────────────────
def create_tables():
    """Create all tables if they don't exist. Raises on connection failure."""
    try:
        print("[DB] Creating tables...")
        Base.metadata.create_all(engine)
        print("[DB] ✅ Tables created successfully")
        return True
    except Exception as e:
        print(f"[DB] ❌ Failed to create tables: {e}")
        raise


# ── Connection Test Helper ─────────────────────────────────────
def test_connection():
    """Test if database is reachable. Useful for diagnostics."""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))  # FIX: raw SQL must be wrapped in text()
        db.close()
        print("[DB] ✅ Connection test passed")
        return True
    except Exception as e:
        print(f"[DB] ❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("[DB] Running standalone table creation...")
    test_connection()
    create_tables()