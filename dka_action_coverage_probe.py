"""
dka_action_coverage_probe.py
============================
Before rewriting the action extractor, find WHERE potassium and insulin actually
live in this MIMIC build: inputevents (icu drips), prescriptions (orders), or
emar (administration records). Counts only, no extraction. Prints sample drug
strings so we can see forms/units to convert later.

Run:  MIMIC_DIR=/path python dka_action_coverage_probe.py
"""

import os
import duckdb

MIMIC_DIR = os.environ.get("MIMIC_DIR", "/path/to/mimic-iv")
HOSP, ICU = os.path.join(MIMIC_DIR, "hosp"), os.path.join(MIMIC_DIR, "icu")


def f(mod, name):
    return f"read_csv_auto('{os.path.join(mod, name)}.csv.gz')"


def try_q(c, label, sql):
    try:
        df = c.execute(sql).df()
        print(f"\n### {label}")
        print(df.to_string(index=False) if len(df) else "  (no rows)")
    except Exception as e:
        print(f"\n### {label}\n  unavailable: {str(e)[:120]}")


def main():
    c = duckdb.connect(); c.execute("PRAGMA threads=4;")

    for drug in ("potassium", "insulin"):
        print("\n" + "=" * 60)
        print(f"WHERE IS '{drug.upper()}'?")
        print("=" * 60)

        # 1. icu inputevents (via d_items label)
        try_q(c, f"inputevents itemids matching '{drug}'", f"""
            SELECT d.itemid, d.label, COUNT(*) AS n_events
            FROM {f(ICU,'inputevents')} ie
            JOIN {f(ICU,'d_items')} d ON ie.itemid = d.itemid
            WHERE lower(d.label) LIKE '%{drug}%'
            GROUP BY 1,2 ORDER BY n_events DESC
        """)

        # 2. hosp prescriptions (orders)
        try_q(c, f"prescriptions matching '{drug}' (top drug strings)", f"""
            SELECT drug, dose_unit_rx, route, COUNT(*) AS n
            FROM {f(HOSP,'prescriptions')}
            WHERE lower(drug) LIKE '%{drug}%'
            GROUP BY 1,2,3 ORDER BY n DESC LIMIT 12
        """)

        # 3. hosp emar (administration records)
        try_q(c, f"emar matching '{drug}' (top medications)", f"""
            SELECT medication, event_txt, COUNT(*) AS n
            FROM {f(HOSP,'emar')}
            WHERE lower(medication) LIKE '%{drug}%'
            GROUP BY 1,2 ORDER BY n DESC LIMIT 12
        """)


if __name__ == "__main__":
    main()
