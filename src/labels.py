import argparse
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

LOG = logging.getLogger("labels")


def parse_args():
    p = argparse.ArgumentParser(description="Rule-based label generation for fake job ads")
    p.add_argument("--input", "-i", default="data/india_job_market_dataset_final_modified.xlsx", help="Input Excel or CSV file")
    p.add_argument("--out-labelled", default="data/labelled_jobs.csv", help="Output full labelled CSV")
    p.add_argument("--out-clean", default="data/cleaned_jobs.csv", help="Output compact cleaned CSV")
    p.add_argument("--out-rule-counts", default="data/rule_trigger_counts.csv", help="Output rule counts CSV")
    p.add_argument("--desc-threshold", type=int, default=15, help="Wordcount threshold to consider description 'missing/short'")
    p.add_argument("--min-flags", type=int, default=1, help="Minimum number of flags required to mark fake")
    return p.parse_args()


# ---------- helpers ----------
def parse_salary_range(s):
    """Parse salary strings like '5-8 LPA' or '20+ LPA' returning (lower, upper) in LPA floats."""
    if pd.isna(s) or str(s).strip() == "":
        return (None, None)
    s0 = str(s).lower().replace(" ", "")
    try:
        if "lpa" in s0:
            s1 = s0.replace("lpa", "")
            if "+" in s1:
                v = float(s1.replace("+", ""))
                return (v, v * 2.0)
            if "-" in s1:
                a, b = s1.split("-")
                return (float(a), float(b))
            return (float(s1), float(s1))
        nums = re.findall(r"\d+\.?\d*", s0)
        if len(nums) == 1:
            return (float(nums[0]), float(nums[0]))
        if len(nums) >= 2:
            return (float(nums[0]), float(nums[1]))
    except Exception:
        return (None, None)
    return (None, None)


def extract_experience(exp):
    if pd.isna(exp) or str(exp).strip() == "":
        return None
    m = re.findall(r"\d+", str(exp))
    return int(m[0]) if m else None


def company_size_bucket(s):
    s = str(s).lower()
    if "small" in s or ("1" in s and "50" in s):
        return "Small"
    if "large" in s or "500" in s:
        return "Large"
    if "medium" in s or ("50" in s and "500" in s):
        return "Medium"
    nums = re.findall(r"\d+", s)
    if nums:
        try:
            n = int(nums[0])
            if n <= 50:
                return "Small"
            if n <= 500:
                return "Medium"
            return "Large"
        except Exception:
            return "Unknown"
    return "Unknown"


def email_domain(e):
    try:
        e = str(e).strip().lower()
        return e.split("@")[-1] if "@" in e else ""
    except Exception:
        return ""


# ---------- pipeline ----------
def generate_labels(df, desc_threshold=15, min_flags=1):
    # derived columns
    if "Salary Range" in df.columns:
        df["salary_lower_lpa"], df["salary_upper_lpa"] = zip(*df["Salary Range"].map(parse_salary_range))
    else:
        df["salary_lower_lpa"], df["salary_upper_lpa"] = (pd.Series([None]*len(df)), pd.Series([None]*len(df)))

    df["exp_years"] = df.get("Experience Required", pd.Series([None]*len(df))).map(extract_experience)
    df["desc_wordcount"] = df.get("Job Description", pd.Series([""]*len(df))).map(lambda x: len(str(x).split()))
    df["company_size_bucket"] = df.get("Company Size", pd.Series(["Unknown"]*len(df))).map(company_size_bucket)
    df["email_domain"] = df.get("Recruiter Email", pd.Series([""]*len(df))).map(email_domain)

    # flags
    suspicious_email_domains = {"gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "rediffmail.com", "ymail.com"}
    vague_company_keywords = ["confidential", "notdisclosed", "not disclosed", "unknown", "n/a", "na", "-", "services", "consulting", "consultancy", "hr"]
    trigger_keywords = [
        "work from home fee", "registration fee", "urgent requirement", "limited seats", "part time earn",
        "without interview", "no interview", "no experience required", "apply now", "earn from home",
        "fast earning", "quick join", "send cv to", "whatsapp", "contact whatsapp", "cash in hand"
    ]

    df["flag_email_suspicious"] = df["email_domain"].isin(suspicious_email_domains)
    df["flag_missing_description"] = df.get("Job Description", pd.Series([""]*len(df))).isna() | (df["desc_wordcount"] <= desc_threshold)
    df["flag_vague_company"] = df.get("Company Name", pd.Series([""]*len(df))).astype(str).str.lower().apply(
        lambda x: any(k in x for k in vague_company_keywords) or (len(str(x).strip()) < 3)
    )
    df["flag_salary_unrealistic"] = (
        df["salary_lower_lpa"].apply(lambda x: True if (pd.notna(x) and x < 0.5) else False)
        | df["salary_upper_lpa"].apply(lambda x: True if (pd.notna(x) and x > 60) else False)
    )
    df["flag_location_missing"] = df.get("Job Location", pd.Series([""]*len(df))).astype(str).str.lower().isin(["", "nan", "none", "india"]) | df.get("Job Location", pd.Series([None]*len(df))).isna()
    df["flag_location_multiple_inconsistent"] = df.apply(
        lambda r: True if ("multiple" in str(r.get("Job Location", "")).lower() and "onsite" in str(r.get("Remote/Onsite", "")).lower()) else False,
        axis=1
    )

    df["Number of Applicants"] = pd.to_numeric(df.get("Number of Applicants", pd.Series([np.nan] * len(df))), errors="coerce")
    df["flag_applicants_anomaly"] = df["Number of Applicants"].apply(lambda x: True if pd.notna(x) and x > 1000 else False)

    df["flag_company_salary_mismatch"] = df.apply(lambda r: True if (r.get("company_size_bucket") == "Small" and pd.notna(r.get("salary_upper_lpa")) and r.get("salary_upper_lpa") > 30) else False, axis=1)

    def contains_trigger(text):
        txt = str(text).lower()
        for kw in trigger_keywords:
            if kw in txt:
                return True
        return False

    df["flag_trigger_keyword"] = df.get("Job Description", pd.Series([""]*len(df))).apply(contains_trigger)

    # collect flags
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["num_flags"] = df[flag_cols].sum(axis=1)
    df["fake_label"] = (df["num_flags"] >= min_flags).astype(int)

    def triggered_rules(row):
        return ",".join([c.replace("flag_", "") for c in flag_cols if row.get(c)]) or "none"

    df["triggered_rules"] = df.apply(triggered_rules, axis=1)

    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        LOG.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    # read
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    LOG.info("Rows loaded: %d", len(df))
    df = df.drop_duplicates().reset_index(drop=True)
    LOG.info("Rows after dedup: %d", len(df))

    df_labeled = generate_labels(df, desc_threshold=args.desc_threshold, min_flags=args.min_flags)

    # save outputs
    out_labelled = Path(args.out_labelled)
    out_clean = Path(args.out_clean)
    out_rule_counts = Path(args.out_rule_counts)
    out_labelled.parent.mkdir(parents=True, exist_ok=True)
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_rule_counts.parent.mkdir(parents=True, exist_ok=True)

    df_labeled.to_csv(out_labelled, index=False)

    cols_to_keep = ["Job ID", "Job Title", "Company Name", "Job Location", "Salary Range", "Experience Required", "Job Portal", "Number of Applicants", "Company Size", "Job Description", "Recruiter Email", "fake_label", "triggered_rules"]
    cols_present = [c for c in cols_to_keep if c in df_labeled.columns]
    df_labeled[cols_present].to_csv(out_clean, index=False)

    flag_cols = [c for c in df_labeled.columns if c.startswith("flag_")]
    rule_counts = df_labeled[flag_cols].sum().sort_values(ascending=False).reset_index()
    rule_counts.columns = ["rule", "count"]
    rule_counts.to_csv(out_rule_counts, index=False)

    LOG.info("Saved labelled to: %s", out_labelled)
    LOG.info("Saved clean to: %s", out_clean)
    LOG.info("Saved rule counts to: %s", out_rule_counts)

    total = len(df_labeled)
    flagged = int(df_labeled["fake_label"].sum())
    LOG.info("Total rows processed: %d", total)
    LOG.info("Rows labelled fake: %d (%.2f%%)", flagged, 100.0 * flagged / max(1, total))


if __name__ == "__main__":
    main()
