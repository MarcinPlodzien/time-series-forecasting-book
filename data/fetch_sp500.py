"""
fetch_sp500.py
=============

Download the S&P 500 daily closing series used by the book and write it where the
case study and the benchmark expect it. The S&P series is NOT redistributed in
this repository (its terms are unclear), so it is fetched on demand instead.

Source: Stooq's free daily history for the S&P 500 index (^SPX),
    https://stooq.com/q/d/l/?s=^spx&i=d
which returns CSV columns Date,Open,High,Low,Close,Volume. We keep the Close
column, one value per line, oldest first.

    python data/fetch_sp500.py

Writes:
    data/sp500/sp500_daily_close.txt          (case study, Chapter 3)
    data/benchmarks/sp500_daily_close.txt     (benchmark, Chapter 9)

Note on reproducibility: the exact series in the book had 24671 rows
(1927-2026) with sha256 7af3f592bdd96d3937791592c54ed313f7895920a0e5640d1dc2b13150ef0167.
A fresh download will differ in vintage/coverage and will NOT match that hash
byte-for-byte; that is expected. The script prints the row count and sha256 of
what it fetched so you can compare. Results are qualitatively identical (the S&P
returns are unforecastable either way -- that is the point of the example).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

URL = "https://stooq.com/q/d/l/?s=^spx&i=d"
REF_SHA256 = "7af3f592bdd96d3937791592c54ed313f7895920a0e5640d1dc2b13150ef0167"
REF_ROWS = 24671

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [
    os.path.join(HERE, "sp500", "sp500_daily_close.txt"),
    os.path.join(HERE, "benchmarks", "sp500_daily_close.txt"),
]


def main() -> None:
    print(f"downloading S&P 500 daily history from {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].lower().startswith("date"):
        raise SystemExit(f"unexpected response (first line: {lines[:1]}); "
                         "Stooq may rate-limit -- try again shortly.")
    closes = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) >= 5:
            try:
                closes.append(float(parts[4]))  # Close column
            except ValueError:
                pass
    if not closes:
        raise SystemExit("no Close values parsed; check the source format.")

    body = "\n".join(f"{c:.6f}" for c in closes) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    for path in TARGETS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(body)
        print(f"  wrote {path}")
    print(f"rows={len(closes)}  sha256={digest}")
    if len(closes) == REF_ROWS and digest == REF_SHA256:
        print("matches the book's reference series exactly.")
    else:
        print(f"differs from the book's reference (rows={REF_ROWS}, "
              f"sha256={REF_SHA256[:12]}...); this is expected for a fresh "
              "download and does not change the conclusions.")


if __name__ == "__main__":
    main()
