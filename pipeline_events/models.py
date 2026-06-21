# pipeline/models.py
import os
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://polyglot:polyglot_pass@localhost:5432/polyglot_echo"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(String)
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
    created_at      = Column(DateTime, default=datetime.utcnow)

class StageEvent(Base):
    __tablename__ = "stage_events"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String)
    event_type    = Column(String(50))
    latency_ms    = Column(Integer)
    metadata_json = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(engine)
    print("[DB] Tables created successfully")

if __name__ == "__main__":
    create_tables()