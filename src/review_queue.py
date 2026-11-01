import sqlite3
from pathlib import Path

DB_PATH = Path("data/moderation_queue.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            confidence REAL,
            prediction INTEGER,
            explanation TEXT,
            status TEXT DEFAULT 'PENDING',
            moderator_correction INTEGER DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_to_queue(text: str, confidence: float, prediction: int, explanation: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO review_queue (text, confidence, prediction, explanation, status)
        VALUES (?, ?, ?, ?, 'PENDING')
    """, (text, confidence, prediction, explanation))
    conn.commit()
    conn.close()

def get_pending_items():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, confidence, prediction, explanation FROM review_queue WHERE status = 'PENDING'")
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_moderator_feedback(item_id: int, correction: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE review_queue 
        SET moderator_correction = ?, status = 'REVIEWED'
        WHERE id = ?
    """, (correction, item_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("SQLite moderation review queue initialized successfully!")