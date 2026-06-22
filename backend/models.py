# pipeline/models.py
import os
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text, pool
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
# This is optimized for Railway managed PostgreSQL
try:
    engine = create_engine(
        DATABASE_URL,
        # Connection pool settings:
        # - NullPool: No connection pooling (good for serverless/Railway)
        # - pool_pre_ping: Test connections before using them
        poolclass=pool.NullPool,
        pool_pre_ping=True,
        # Timeouts:
        connect_args={
            "connect_timeout": 10,  # 10 second timeout to connect
            "application_name": "polyglot_echo"
        },
        echo=False  # Set to True for SQL debug logging
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
    session_id      = Column(String, index=True)  # Add index for faster queries
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
    session_id    = Column(String, index=True)  # Add index
    event_type    = Column(String(50))
    latency_ms    = Column(Integer)
    metadata_json = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)

# ── Table Creation with Error Handling ──────────────────────────
def create_tables():
    """
    Create all tables if they don't exist.
    Raises exception if connection fails.
    """
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
    """
    Test if database is reachable.
    Useful for diagnostics.
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
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