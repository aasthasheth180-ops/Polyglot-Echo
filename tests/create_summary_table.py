# create_summary_table.py
import sqlalchemy as sa

DB_URL = "postgresql://polyglot:polyglot_pass@localhost:5432/polyglot_echo"
engine = sa.create_engine(DB_URL)

with engine.connect() as conn:
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id              SERIAL PRIMARY KEY,
            summary_date    DATE NOT NULL UNIQUE,
            total_turns     INTEGER DEFAULT 0,
            avg_whisper_ms  FLOAT DEFAULT 0,
            avg_llm_ms      FLOAT DEFAULT 0,
            avg_tts_ms      FLOAT DEFAULT 0,
            avg_total_ms    FLOAT DEFAULT 0,
            cache_hit_rate  FLOAT DEFAULT 0,
            lang_en_count   INTEGER DEFAULT 0,
            lang_hi_count   INTEGER DEFAULT 0,
            lang_gu_count   INTEGER DEFAULT 0,
            tts_drift_pct   FLOAT DEFAULT 0,
            drift_alert     BOOLEAN DEFAULT FALSE,
            created_at      TIMESTAMP DEFAULT NOW()
        )
    """))
    conn.commit()
    print("✅ Success: daily_summaries analytics table structural schema created!")