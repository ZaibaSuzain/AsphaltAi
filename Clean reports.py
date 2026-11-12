"""
clean_reports.py - remove saved reports that have no locality or a location outside Bengaluru.

Run it from the project folder (with the venv active):
    python clean_reports.py

It first SHOWS which reports it would delete and asks you to type "yes" before removing
anything. Reports with a locality and a location inside Bengaluru are never touched,
including the sample reports that come with the app.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path("potholeiq.db")                   # same database file the app uses
SOUTH, NORTH, WEST, EAST = 12.7, 13.3, 77.3, 77.9   # same limits as BENGALURU_BOUNDS in app.py

FIND_BAD_REPORTS = """
    SELECT id, created_at, locality, latitude, longitude
    FROM reports
    WHERE locality IS NULL OR TRIM(locality) = ''
       OR latitude NOT BETWEEN ? AND ?
       OR longitude NOT BETWEEN ? AND ?
    ORDER BY id DESC
"""


def main():
    if not DB_PATH.exists():
        raise SystemExit(f"Database file not found: {DB_PATH}\n"
                         "Run this from the project folder, the one that contains app.py.")

    with closing(sqlite3.connect(DB_PATH)) as conn:
        bad = conn.execute(FIND_BAD_REPORTS, (SOUTH, NORTH, WEST, EAST)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]

        if not bad:
            print(f"All {total} reports have a locality and a location inside Bengaluru. "
                  "Nothing to remove.")
            return

        print(f"{len(bad)} of {total} reports have a missing locality or a location "
              "outside Bengaluru:\n")
        print(f"{'id':>4}  {'saved':16}  {'locality':28}  {'latitude':>9}  {'longitude':>9}")
        for report_id, created_at, locality, latitude, longitude in bad:
            shown = (locality or "(none)")[:28]
            print(f"{report_id:>4}  {created_at:16}  {shown:28}  {latitude:9.4f}  {longitude:9.4f}")

        answer = input("\nDelete these reports? Type yes to confirm: ").strip().lower()
        if answer != "yes":
            print("Nothing was deleted.")
            return

        conn.executemany("DELETE FROM reports WHERE id = ?", [(row[0],) for row in bad])
        conn.commit()
        print(f"Deleted {len(bad)} reports. {total - len(bad)} remain.")


if __name__ == "__main__":
    main()