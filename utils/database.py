"""
database.py
-----------
SQLAlchemy models and engine for the Face Recognition Attendance System.
Uses SQLite stored at data/attendance.db.
"""

import os
import json
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Date,
    UniqueConstraint, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
JSON_PATH = os.path.join(DATA_DIR, "attendance.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Engine & Session
# ─────────────────────────────────────────────────────────────────────────────
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    image_count = Column(Integer, default=0)

    def __repr__(self):
        return f"<Student(name='{self.name}', images={self.image_count})>"


class AttendanceRecord(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    date = Column(String, nullable=False, index=True)   # YYYY-MM-DD
    time = Column(String, nullable=False)                # HH:MM:SS
    confidence = Column(Float, default=0.0)

    # Prevent duplicate (name, date) entries
    __table_args__ = (
        UniqueConstraint("name", "date", name="uq_name_date"),
    )

    def __repr__(self):
        return f"<Attendance(name='{self.name}', date='{self.date}')>"

    def to_dict(self):
        return {
            "name": self.name,
            "date": self.date,
            "time": self.time,
            "confidence": self.confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Create tables
# ─────────────────────────────────────────────────────────────────────────────
Base.metadata.create_all(engine)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-migrate from JSON
# ─────────────────────────────────────────────────────────────────────────────
def migrate_from_json():
    """
    One-time migration: import records from attendance.json into SQLite.
    Only runs if the JSON file exists and the DB table is empty.
    """
    if not os.path.exists(JSON_PATH):
        return

    session = SessionLocal()
    try:
        count = session.query(AttendanceRecord).count()
        if count > 0:
            return  # DB already has data, skip migration

        with open(JSON_PATH, "r") as f:
            records = json.load(f)

        if not isinstance(records, list):
            return

        migrated = 0
        for rec in records:
            try:
                ar = AttendanceRecord(
                    name=rec.get("name", ""),
                    date=rec.get("date", ""),
                    time=rec.get("time", ""),
                    confidence=float(rec.get("confidence", 0.0)),
                )
                session.add(ar)
                session.commit()
                migrated += 1
            except Exception:
                session.rollback()

        if migrated > 0:
            # Rename the old JSON to mark it as migrated
            backup = JSON_PATH + ".migrated"
            os.rename(JSON_PATH, backup)
            print(f"✅ Migrated {migrated} records from JSON → SQLite. Old file renamed to {backup}")
    finally:
        session.close()


# Run migration on import
migrate_from_json()
