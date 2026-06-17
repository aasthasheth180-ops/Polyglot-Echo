# pipeline/dags/daily_analytics_runner.py
import sqlalchemy as sa
from datetime import datetime, timedelta, date

DB_CONN = "postgresql://polgyglot:polyglot_pass@localhost:5432/polyglot_echo"

def compute_stats() -> dict:
    """Stage 1: Aggregate performance metrics from conversation history logs"""
    engine = sa.create_engine(DB_CONN)
    yesterday = date.today() - timedelta(days=1)
    
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT
                COUNT(*)                                        AS total_turns,
                AVG(whisper_ms)                                 AS avg_whisper_ms,
                AVG(llm_ms)                                     AS avg_llm_ms,
                AVG(tts_ms)                                     AS avg_tts_ms,
                AVG(total_ms)                                   AS avg_total_ms,
                AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END)  AS cache_hit_rate,
                SUM(CASE WHEN output_lang='EN' THEN 1 ELSE 0 END) AS lang_en,
                SUM(CASE WHEN output_lang='HI' THEN 1 ELSE 0 END) AS lang_hi,
                SUM(CASE WHEN output_lang='GU' THEN 1 ELSE 0 END) AS lang_gu
            FROM conversation_turns
            WHERE DATE(created_at) = :d
        """), {"d": yesterday}).fetchone()
        
    stats = {
        "summary_date": str(yesterday),
        "total_turns": int(row.total_turns or 0),
        "avg_whisper_ms": float(row.avg_whisper_ms or 0),
        "avg_llm_ms": float(row.avg_llm_ms or 0),
        "avg_tts_ms": float(row.avg_tts_ms or 0),
        "avg_total_ms": float(row.avg_total_ms or 0),
        "cache_hit_rate": float(row.cache_hit_rate or 0),
        "lang_en": int(row.lang_en or 0),
        "lang_hi": int(row.lang_hi or 0),
        "lang_gu": int(row.lang_gu or 0),
    }
    print(f"📊 [ETL Task 1/3] Computed stats for {yesterday}: {stats['total_turns']} logs detected.")
    return stats

def detect_drift(stats: dict) -> dict:
    """Stage 2: Evaluate model degradation by analyzing rolling window variations"""
    engine = sa.create_engine(DB_CONN)
    
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT AVG(avg_tts_ms) AS rolling_avg
            FROM daily_summaries
            WHERE summary_date >= CURRENT_DATE - INTERVAL '7 days'
        """)).fetchone()
        
    rolling = float(row.rolling_avg or 0)
    today = stats["avg_tts_ms"]
    
    drift = ((today - rolling) / rolling * 100) if rolling > 0 else 0
    stats["drift_pct"] = round(drift, 2)
    stats["drift_alert"] = drift > 20.0
    
    print(f"📉 [ETL Task 2/3] Drift Evaluation complete: Current drift is {drift:+.1f}%")
    if stats["drift_alert"]:
        print(f"⚠️ [DRIFT ALERT] Voice Synthesis module experiencing performance degradation!")
    return stats

def write_summary(stats: dict):
    """Stage 3: Upsert metrics directly into analytical data structures"""
    engine = sa.create_engine(DB_CONN)
    
    with engine.connect() as conn:
        conn.execute(sa.text("""
            INSERT INTO daily_summaries
                (summary_date, total_turns, avg_whisper_ms, avg_llm_ms, 
                 avg_tts_ms, avg_total_ms, cache_hit_rate, 
                 lang_en_count, lang_hi_count, lang_gu_count, 
                 tts_drift_pct, drift_alert)
            VALUES
                (:summary_date, :total_turns, :avg_whisper_ms, :avg_llm_ms, 
                 :avg_tts_ms, :avg_total_ms, :cache_hit_rate, 
                 :lang_en, :lang_hi, :lang_gu, 
                 :drift_pct, :drift_alert)
            ON CONFLICT (summary_date) DO UPDATE SET
                total_turns = EXCLUDED.total_turns,
                avg_total_ms = EXCLUDED.avg_total_ms,
                tts_drift_pct = EXCLUDED.tts_drift_pct,
                drift_alert = EXCLUDED.drift_alert
        """), stats)
        conn.commit()
    print(f"💾 [ETL Task 3/3] Successfully compiled and committed to daily_summaries table!\n")

# Main execution trigger entry point
if __name__ == "__main__":
    print("🚀 Simulating Polyglot Echo Analytical DAG Execution Window...")
    stage_1 = compute_stats()
    stage_2 = detect_drift(stage_1)
    write_summary(stage_2)
    print("🏁 Execution array complete using optimized local processing resources.")