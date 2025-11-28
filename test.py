# run_test.py — quick checks for dataset & labels
import pandas as pd, os, sys
p = "data/india_job_market_dataset_final_modified.xlsx"

print("Checking input file:", p)
if not os.path.exists(p):
    print("ERROR: input file not found at", p)
    sys.exit(1)

try:
    df = pd.read_excel(p)
    print("Loaded OK — rows:", len(df))
    print("Columns:", df.columns.tolist()[:40])
    # print first 2 job descriptions (safe length)
    if "Job Description" in df.columns:
        print("\nSample Job Description (first 2 rows):")
        for i, txt in enumerate(df["Job Description"].head(2).astype(str)):
            print(f"--- row {i+1} ({len(txt.split())} words) ---")
            print(txt[:1000])   # print up to first 1000 chars
    else:
        print("No 'Job Description' column found.")
except Exception as e:
    print("ERROR loading file:", type(e).__name__, e)
    import traceback; traceback.print_exc()
    sys.exit(1)
