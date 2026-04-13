"""
attendance.py
-------------
Manages attendance records stored in SQLite via SQLAlchemy.
Provides mark, query, and export helpers.
"""

from datetime import datetime
from typing import Optional
from collections import OrderedDict

from sqlalchemy import func
from utils.database import SessionLocal, AttendanceRecord

# Date / time format strings
DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H:%M:%S"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def mark_attendance(name: str, confidence: float) -> tuple[bool, str]:
    """
    Record attendance for *name* with the given *confidence* score.

    Duplicate entries (same name + same calendar date) are silently skipped.

    Returns
    -------
    (marked: bool, message: str)
      marked=True  → new record written
      marked=False → duplicate detected, nothing written
    """
    if name == "Unknown":
        return False, "Unknown face — not recorded."

    now = datetime.now()
    today = now.strftime(DATE_FMT)
    current_time = now.strftime(TIME_FMT)

    session = SessionLocal()
    try:
        # Duplicate check
        existing = (
            session.query(AttendanceRecord)
            .filter_by(name=name, date=today)
            .first()
        )
        if existing:
            return False, f"'{name}' already marked present today."

        record = AttendanceRecord(
            name=name,
            date=today,
            time=current_time,
            confidence=round(float(confidence), 4),
        )
        session.add(record)
        session.commit()
        return True, f"✅ Attendance marked for '{name}' at {current_time}."
    except Exception as e:
        session.rollback()
        return False, f"Database error: {e}"
    finally:
        session.close()


def get_attendance_records(
    name_filter: Optional[str] = None,
    date_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict]:
    """
    Return all attendance records, optionally filtered.

    Parameters
    ----------
    name_filter : If given, only records whose name contains this string
                  (case-insensitive) are returned.
    date_filter : If given (YYYY-MM-DD), only records for that date are returned.
    date_from   : Start of date range (inclusive).
    date_to     : End of date range (inclusive).
    """
    session = SessionLocal()
    try:
        query = session.query(AttendanceRecord)

        if name_filter:
            query = query.filter(AttendanceRecord.name.ilike(f"%{name_filter}%"))
        if date_filter:
            query = query.filter(AttendanceRecord.date == date_filter)
        if date_from:
            query = query.filter(AttendanceRecord.date >= date_from)
        if date_to:
            query = query.filter(AttendanceRecord.date <= date_to)

        query = query.order_by(AttendanceRecord.date, AttendanceRecord.time)
        return [r.to_dict() for r in query.all()]
    finally:
        session.close()


def get_today_attendance() -> list[dict]:
    """Return today's attendance records."""
    today = datetime.now().strftime(DATE_FMT)
    return get_attendance_records(date_filter=today)


def get_present_today() -> list[str]:
    """Return a list of names that have been marked present today."""
    return [r["name"] for r in get_today_attendance()]


def get_daily_counts() -> dict[str, int]:
    """
    Return a mapping of {date_string: count} for all recorded dates.
    Useful for charting attendance trends.
    """
    session = SessionLocal()
    try:
        results = (
            session.query(AttendanceRecord.date, func.count(AttendanceRecord.id))
            .group_by(AttendanceRecord.date)
            .order_by(AttendanceRecord.date)
            .all()
        )
        return OrderedDict(results)
    finally:
        session.close()


def get_student_attendance_history(name: str) -> list[dict]:
    """Return all attendance records for a specific student."""
    return get_attendance_records(name_filter=name)


def get_weekly_summary(date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
    """
    Return summary statistics for a date range.
    """
    records = get_attendance_records(date_from=date_from, date_to=date_to)
    if not records:
        return {
            "total_records": 0,
            "unique_students": 0,
            "avg_confidence": 0.0,
            "unique_dates": 0,
        }
    
    names = set(r["name"] for r in records)
    dates = set(r["date"] for r in records)
    avg_conf = sum(float(r["confidence"]) for r in records) / len(records)
    
    return {
        "total_records": len(records),
        "unique_students": len(names),
        "avg_confidence": round(avg_conf, 4),
        "unique_dates": len(dates),
    }


def get_student_stats(name: str) -> dict:
    """
    Return detailed stats for a single student.
    """
    records = get_attendance_records(name_filter=name)
    if not records:
        return {
            "total_days": 0,
            "first_seen": None,
            "last_seen": None,
            "avg_confidence": 0.0,
            "streak": 0,
        }

    dates = sorted(set(r["date"] for r in records))
    avg_conf = sum(float(r["confidence"]) for r in records) / len(records)

    # Calculate current streak
    from datetime import timedelta
    streak = 1
    for i in range(len(dates) - 1, 0, -1):
        d1 = datetime.strptime(dates[i], DATE_FMT)
        d2 = datetime.strptime(dates[i - 1], DATE_FMT)
        if (d1 - d2).days == 1:
            streak += 1
        else:
            break

    return {
        "total_days": len(dates),
        "first_seen": dates[0],
        "last_seen": dates[-1],
        "avg_confidence": round(avg_conf, 4),
        "streak": streak,
    }


def delete_student_records(name: str) -> int:
    """
    Delete all attendance records for a student.
    Returns the number of records deleted.
    """
    session = SessionLocal()
    try:
        count = session.query(AttendanceRecord).filter_by(name=name).delete()
        session.commit()
        return count
    except Exception:
        session.rollback()
        return 0
    finally:
        session.close()

