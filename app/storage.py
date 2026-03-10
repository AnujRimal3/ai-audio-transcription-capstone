"""
storage.py
----------

SQLite database management for LocalTranscriber.

This module handles creation and access to the application's
persistent job tracking database.

Responsibilities
----------------
- Create database connection
- Initialize schema
- Store job metadata
- Track processing status
- Store artifact locations
- Record error messages

Database Tables
---------------
jobs

Fields:
    id                   Unique job identifier (UUID)
    created_at           Job creation timestamp
    input_path           Original media file path
    status               running / done / error
    transcript_json_path Transcript output location
    summary_md_path      Summary output location
    error                Error message if job failed

Course: C486 - Senior Capstone Computer Science
Team 6

Team Members:
- Anuj Rimal
- Hoyda Al Yahiri
- Zack Ganser
- Saahil Patel

Project:
AI Audio Transcription & Summarization System
"""

from __future__ import annotations
import sqlite3
from app.paths import user_data_dir

# ---------------------------------------------------------
# Database path
# ---------------------------------------------------------
# The SQLite database file used by the application.
# It is stored inside the user's application data directory so that:
# - it persists across app launches
# - it is not mixed with program files
# - each user has their own database
DB_PATH = user_data_dir() / "app.db"

# ---------------------------------------------------------
# Create database connection
# ---------------------------------------------------------
def connect() -> sqlite3.Connection:
    """
    Open a connection to the application's SQLite database.

    Setting `row_factory = sqlite3.Row` allows queries to return rows
    that can be accessed like:

        row["column_name"]

    instead of numeric index access like:

        row[0]
    """

    # Open connection to database file
    conn = sqlite3.connect(DB_PATH)

    # Configure row results to behave like dictionaries
    conn.row_factory = sqlite3.Row

    return conn

# ---------------------------------------------------------
# Initialize database schema
# ---------------------------------------------------------
def init_db() -> None:
    """
    Initialize the database schema if it does not already exist.

    This function creates the `jobs` table which stores metadata for
    transcription + summarization jobs executed by the application.

    The table tracks:
    - job id
    - creation timestamp
    - original input file
    - processing status
    - output artifact paths
    - any error messages

    This function is safe to call multiple times because the SQL
    statement uses `CREATE TABLE IF NOT EXISTS`. -Zack
    """

    # Open database connection using context manager
    # (automatically commits/rolls back on exit)
    with connect() as conn:
        # Create jobs table if it doesn't already exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            input_path TEXT NOT NULL,
            status TEXT NOT NULL,
            transcript_json_path TEXT,
            summary_md_path TEXT,
            error TEXT
        )
        """)

        # Explicitly commit schema creation
        conn.commit()