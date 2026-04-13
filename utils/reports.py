"""
reports.py
----------
PDF report generation for attendance data using fpdf2.
"""

from datetime import datetime
from fpdf import FPDF

from utils.attendance import get_attendance_records, get_weekly_summary


class AttendanceReport(FPDF):
    """Custom PDF class with header/footer branding."""

    def __init__(self, title="Attendance Report"):
        super().__init__()
        self._title = title

    def header(self):
        self.set_fill_color(15, 32, 39)
        self.rect(0, 0, 210, 30, "F")
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.set_y(8)
        self.cell(0, 10, f"  {self._title}", align="L")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(160, 174, 192)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="R")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf_report(
    name_filter=None,
    date_from=None,
    date_to=None,
) -> bytes:
    """
    Generate a branded PDF attendance report.

    Returns the PDF as bytes (suitable for st.download_button).
    """
    records = get_attendance_records(
        name_filter=name_filter,
        date_from=date_from,
        date_to=date_to,
    )
    summary = get_weekly_summary(date_from=date_from, date_to=date_to)

    # Determine title
    title_parts = ["Attendance Report"]
    if name_filter:
        title_parts.append(f"— {name_filter}")
    if date_from and date_to:
        title_parts.append(f"({date_from} to {date_to})")
    elif date_from:
        title_parts.append(f"(from {date_from})")
    elif date_to:
        title_parts.append(f"(until {date_to})")

    pdf = AttendanceReport(title=" ".join(title_parts))
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Summary Box ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(44, 83, 100)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)

    pdf.set_fill_color(240, 245, 250)
    pdf.set_draw_color(99, 179, 237)

    summary_data = [
        ("Total Records", str(summary["total_records"])),
        ("Unique Students", str(summary["unique_students"])),
        ("Days Covered", str(summary["unique_dates"])),
        ("Avg. Confidence", f"{summary['avg_confidence']:.1%}"),
    ]

    col_w = 47
    for label, value in summary_data:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(col_w, 8, value, border=0, align="C", fill=True)
    pdf.ln()
    for label, value in summary_data:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(col_w, 6, label, border=0, align="C")
    pdf.ln(12)

    # ── Records Table ────────────────────────────────────────────────────────
    if not records:
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 10, "No records match the given filters.", ln=True)
        return bytes(pdf.output())

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(44, 83, 100)
    pdf.cell(0, 10, f"Attendance Records ({len(records)} entries)", ln=True)
    pdf.ln(2)

    # Table header
    headers = ["#", "Name", "Date", "Time", "Confidence"]
    col_widths = [12, 60, 35, 30, 35]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(15, 32, 39)
    pdf.set_text_color(255, 255, 255)
    for header, w in zip(headers, col_widths):
        pdf.cell(w, 8, header, border=0, align="C", fill=True)
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 9)
    for i, rec in enumerate(records):
        if i % 2 == 0:
            pdf.set_fill_color(248, 250, 252)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_text_color(50, 50, 50)
        pdf.cell(col_widths[0], 7, str(i + 1), border=0, align="C", fill=True)
        pdf.cell(col_widths[1], 7, rec["name"], border=0, align="L", fill=True)
        pdf.cell(col_widths[2], 7, rec["date"], border=0, align="C", fill=True)
        pdf.cell(col_widths[3], 7, rec["time"], border=0, align="C", fill=True)
        pdf.cell(col_widths[4], 7, f"{float(rec['confidence']):.1%}", border=0, align="C", fill=True)
        pdf.ln()

    return bytes(pdf.output())
