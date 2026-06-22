#!/usr/bin/env python3
"""
🔍 Polyglot Echo Database Diagnostics Script

Run this BEFORE deploying to Railway to catch connection issues early.
It will test:
1. DATABASE_URL environment variable
2. PostgreSQL connection
3. Table creation
4. Query operations

Usage:
    python test_railway_db.py
"""

import os
import sys
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def print_success(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_warning(msg):
    print(f"⚠️  {msg}")

def test_env_variable():
    """Test 1: Check DATABASE_URL is set"""
    print_header("Test 1: Environment Variable")
    
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print_error("DATABASE_URL not set in environment")
        print_warning("Make sure your .env file contains: DATABASE_URL=postgresql://...")
        return False
    
    # Mask the password for display
    masked_url = db_url.replace(db_url.split("@")[0].split(":")[-1], "***", 1)
    print_success(f"DATABASE_URL is set")
    print(f"  URL: {masked_url}\n")
    
    # Check format
    if not db_url.startswith("postgresql://"):
        print_error("DATABASE_URL should start with 'postgresql://'")
        return False
    
    print_success("URL format looks correct")
    return True

def test_sqlalchemy_import():
    """Test 2: Check SQLAlchemy is installed"""
    print_header("Test 2: SQLAlchemy Installation")
    
    try:
        from sqlalchemy import create_engine, __version__
        print_success(f"SQLAlchemy {__version__} is installed")
        return True
    except ImportError as e:
        print_error(f"SQLAlchemy not installed: {e}")
        print_warning("Run: pip install sqlalchemy")
        return False

def test_engine_creation():
    """Test 3: Create SQLAlchemy engine"""
    print_header("Test 3: SQLAlchemy Engine Creation")
    
    from sqlalchemy import create_engine
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        engine = create_engine(
            db_url,
            poolclass=__import__('sqlalchemy.pool', fromlist=['NullPool']).NullPool,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10}
        )
        print_success("Engine created successfully")
        return engine
    except Exception as e:
        print_error(f"Failed to create engine: {e}")
        return None

def test_connection(engine):
    """Test 4: Test actual database connection"""
    print_header("Test 4: Database Connection")
    
    if engine is None:
        print_error("Skipping (no engine)")
        return False
    
    try:
        with engine.connect() as conn:
            result = conn.execute(__import__('sqlalchemy', fromlist=['text']).text("SELECT 1"))
            print_success(f"Connected to database")
            print(f"  Query result: {result.fetchone()}\n")
            return True
    except Exception as e:
        print_error(f"Connection failed: {e}")
        print_warning("Common causes:")
        print_warning("  - Wrong password in DATABASE_URL")
        print_warning("  - PostgreSQL not running")
        print_warning("  - Wrong hostname (should be *.railway.app for Railway)")
        print_warning("  - Firewall blocking connection")
        return False

def test_table_creation(engine):
    """Test 5: Create schema and tables"""
    print_header("Test 5: Table Creation")
    
    if engine is None:
        print_error("Skipping (no engine)")
        return False
    
    try:
        from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text
        from sqlalchemy.orm import declarative_base
        from datetime import datetime
        
        Base = declarative_base()
        
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
        
        Base.metadata.create_all(engine)
        print_success("Tables created (or already exist)")
        return True
        
    except Exception as e:
        print_error(f"Failed to create tables: {e}")
        return False

def test_crud_operations(engine):
    """Test 6: Basic CRUD operations"""
    print_header("Test 6: CRUD Operations")
    
    if engine is None:
        print_error("Skipping (no engine)")
        return False
    
    try:
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text
        from sqlalchemy.orm import declarative_base
        from datetime import datetime
        
        Base = declarative_base()
        
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
        
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        # Create test record
        test_turn = ConversationTurn(
            session_id="test_session_123",
            input_text="Hello, how are you?",
            input_lang="en",
            output_text="I'm doing great!",
            output_lang="en",
            speaker_profile="test",
            whisper_ms=100,
            llm_ms=200,
            tts_ms=150,
            total_ms=450,
            cache_hit=False
        )
        db.add(test_turn)
        db.commit()
        print_success("INSERT successful")
        
        # Read test record
        result = db.query(ConversationTurn).filter_by(session_id="test_session_123").first()
        if result:
            print_success(f"SELECT successful: Found record with ID {result.id}")
        
        # Delete test record
        db.query(ConversationTurn).filter_by(session_id="test_session_123").delete()
        db.commit()
        print_success("DELETE successful")
        
        db.close()
        return True
        
    except Exception as e:
        print_error(f"CRUD operations failed: {e}")
        return False

def main():
    """Run all diagnostics"""
    print_header("🔍 Polyglot Echo Database Diagnostics")
    
    results = []
    
    # Test 1
    if not test_env_variable():
        print_error("Cannot proceed without DATABASE_URL")
        sys.exit(1)
    results.append(("Environment Variable", True))
    
    # Test 2
    if not test_sqlalchemy_import():
        print_error("Cannot proceed without SQLAlchemy")
        sys.exit(1)
    results.append(("SQLAlchemy Import", True))
    
    # Test 3
    engine = test_engine_creation()
    results.append(("Engine Creation", engine is not None))
    
    if engine is None:
        print_error("Cannot proceed without working engine")
        sys.exit(1)
    
    # Test 4
    connection_ok = test_connection(engine)
    results.append(("Connection", connection_ok))
    
    if not connection_ok:
        print_error("Cannot proceed without database connection")
        sys.exit(1)
    
    # Test 5
    tables_ok = test_table_creation(engine)
    results.append(("Table Creation", tables_ok))
    
    # Test 6
    crud_ok = test_crud_operations(engine)
    results.append(("CRUD Operations", crud_ok))
    
    # Summary
    print_header("📊 Diagnostics Summary")
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print_success("\nAll diagnostics passed! ✅")
        print_success("Your database is configured correctly.")
        print_success("You can now deploy to Railway with confidence.")
        return 0
    else:
        print_error("\nSome diagnostics failed. ❌")
        print_error("Fix the errors above before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())