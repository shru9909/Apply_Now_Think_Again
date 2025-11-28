import argparse
import logging
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

LOG = logging.getLogger("fe")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------- utils ----------------
def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    # replace urls, emails, numbers
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"\d{4,}", " ", s)            # long numbers (years/ids)
    s = re.sub(r"\d+", " ", s)               # other numbers
    s = re.sub(r"[^\w\s]", " ", s)           # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_salary(s):
    if pd.isna(s): return (np.nan, np.nan)
    s0 = str(s).lower().replace("lpa", "").replace("inr", "").replace("₹", "").strip()
    nums = re.findall(r"\d+\.?\d*", s0)
    if len(nums) == 0: return (np.nan, np.nan)
    if len(nums) == 1:
        v = float(nums[0]); return (v, v)
    return (float(nums[0]), float(nums[1]))

def parse_experience(s):
    if pd.isna(s): return (np.nan, np.nan)
    s0 = str(s).lower()
    nums = re.findall(r"\d+\.?\d*", s0)
    if len(nums) == 0: return (np.nan, np.nan)
    if len(nums) == 1:
        v = float(nums[0]); return (v, v+2)
    return (float(nums[0]), float(nums[1]))

def word_count(text):
    if pd.isna(text): return 0
    return len(re.findall(r"\w+", str(text)))

# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="data/labelled_jobs.csv", help="Labelled CSV (must contain fake_label)")
    p.add_argument("--out-meta", default="data/meta_features.csv")
    p.add_argument("--out-tfidf", default="data/desc_tfidf.npz")
    p.add_argument("--preproc-dir", default="preprocessors")
    p.add_argument("--min-df", type=int, default=3, help="min_df for TF-IDF")
    p.add_argument("--max-features", type=int, default=5000, help="max_features for TF-IDF")
    p.add_argument("--ngram", type=int, default=2, help="max ngram (1 => unigrams, 2 => uni+bi)")
    return p.parse_args()

def main():
    args = parse_args()
    Path(args.preproc_dir).mkdir(parents=True, exist_ok=True)
    LOG.info("Loading input: %s", args.input)
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)

    LOG.info("Rows: %d columns: %d", df.shape[0], df.shape[1])

    # Ensure Job ID exists
    if "Job ID" not in df.columns:
        df["Job ID"] = range(1, len(df) + 1)

    # Ensure fake_label exists
    if "fake_label" not in df.columns:
        LOG.warning("fake_label not found; creating all-zero label (unsafe).")
        df["fake_label"] = 0

    # --- Date features ---
    df["Posted Date"] = pd.to_datetime(df.get("Posted Date"), errors="coerce")
    df["Application Deadline"] = pd.to_datetime(df.get("Application Deadline"), errors="coerce")
    df["posted_year"] = df["Posted Date"].dt.year.fillna(0).astype(int)
    df["posted_month"] = df["Posted Date"].dt.month.fillna(0).astype(int)
    df["posted_dayofweek"] = df["Posted Date"].dt.dayofweek.fillna(0).astype(int)
    df["days_to_deadline"] = (df["Application Deadline"] - df["Posted Date"]).dt.days.fillna(-1).astype(int)
    df["is_urgent"] = (df["days_to_deadline"] <= 7).astype(int)
    df["is_long_term"] = (df["days_to_deadline"] > 30).astype(int)

    # --- Salary ---
    sal = df.get("Salary Range", pd.Series([np.nan] * len(df))).map(parse_salary)
    df["min_salary"] = [x[0] for x in sal]
    df["max_salary"] = [x[1] for x in sal]
    df["avg_salary"] = (df["min_salary"].fillna(0) + df["max_salary"].fillna(0)) / 2

    # --- Experience ---
    exp = df.get("Experience Required", pd.Series([np.nan] * len(df))).map(parse_experience)
    df["min_experience"] = [x[0] for x in exp]
    df["max_experience"] = [x[1] for x in exp]
    df["avg_experience"] = (df["min_experience"].fillna(0) + df["max_experience"].fillna(0)) / 2

    # --- Skills flags ---
    technical_skills = ['python','java','c++','sql','react','aws','machine learning','ui/ux','digital marketing','excel']
    def skills_feat(s):
        s = str(s).lower() if pd.notna(s) else ""
        out = {f"has_{k.replace(' ','_')}": int(k in s) for k in technical_skills}
        toks = [t.strip() for t in s.split(",") if t.strip()]
        out["total_skills"] = len(toks)
        out["technical_skills_count"] = sum(out[f"has_{k.replace(' ','_')}"] for k in technical_skills)
        return out
    skills = df.get("Skills Required", pd.Series([""] * len(df))).map(skills_feat)
    skills_df = pd.DataFrame(list(skills))
    df = pd.concat([df.reset_index(drop=True), skills_df.reset_index(drop=True)], axis=1)

    # --- Description features & cleaned text ---
    df["desc_clean"] = df.get("Job Description", "").map(clean_text)
    df["desc_wordcount"] = df["desc_clean"].map(word_count)
    df["has_remote"] = df["desc_clean"].str.contains(r"\b(remote|work from home)\b").fillna(False).astype(int)
    df["has_urgent"] = df["desc_clean"].str.contains(r"\b(urgent|immediate|start immediately)\b").fillna(False).astype(int)
    df["has_fee"] = df["desc_clean"].str.contains(r"\b(registration fee|pay to apply|fee)\b").fillna(False).astype(int)

    # ---------- India-specific scam signals (novel features) ----------
    # curated India-specific suspicious phrases (expandable)
    india_scam_keywords = [
        "work from home fees", "registration fee", "pay to apply", "earn rs", "earn ₹",
        "earn rs.", "earn rs/-", "earn per week", "no experience earn", "urgent hiring",
        "immediate joining", "start immediately", "earn 50k", "weekly payout", "join and pay",
        "pan india", "anywhere in india", "apply through whatsapp", "send money", "investment required"
    ]

    def india_keyword_count(text):
        t = text.lower() if isinstance(text, str) else ""
        return sum(1 for kw in india_scam_keywords if kw in t)

    # Add counts & boolean flags
    df["india_scam_keyword_count"] = df["desc_clean"].map(india_keyword_count).fillna(0).astype(int)
    df["has_india_scam_kw"] = (df["india_scam_keyword_count"] > 0).astype(int)

    # Suspicious contact channels and missing corporate traceability
    df["mentions_whatsapp"] = df["desc_clean"].str.contains(r"\bwhatsapp\b").fillna(False).astype(int)
    df["mentions_gmail_or_free_email"] = df.get("Recruiter Email", "").astype(str).str.contains(r"(gmail\.com|yahoo\.|hotmail\.)", regex=True).fillna(False).astype(int)
    df["mentions_confidential"] = df.get("Company Name", "").astype(str).str.lower().str.contains(r"\b(confidential|private|na|n/a)\b").fillna(False).astype(int)
    df["has_gst_hint"] = df.get("Company Name", "").astype(str).str.contains(r"\bgst\b", regex=True, case=False).fillna(False).astype(int)

    # Fraud Trigger Index (FTI) - interpretable composite score
    kw_score = df["india_scam_keyword_count"].fillna(0)
    fee_score = df.get("has_fee", pd.Series(0, index=df.index)).fillna(0)
    wa_score = df["mentions_whatsapp"].fillna(0)
    short_desc_score = (df["desc_wordcount"].fillna(0) < 15).astype(int)
    salary_anomaly = ((df["avg_salary"].fillna(0) < 1) | (df["avg_salary"].fillna(0) > 50)).astype(int)

    df["fraud_trigger_index_raw"] = (0.4 * kw_score +
                                     0.25 * fee_score +
                                     0.15 * wa_score +
                                     0.10 * short_desc_score +
                                     0.10 * salary_anomaly)

    mx = df["fraud_trigger_index_raw"].max() if df["fraud_trigger_index_raw"].max() > 0 else 1.0
    df["fraud_trigger_index"] = (df["fraud_trigger_index_raw"] / mx).round(4)
    # -----------------------------------------------------------------

    # --- Location features ---
    major_cities = ['bangalore','mumbai','delhi','hyderabad','chennai','pune','kolkata','ahmedabad','jaipur','noida']
    loc = df.get("Job Location", pd.Series([""] * len(df))).str.lower().fillna("")
    df["is_multiple_locations"] = loc.str.contains("multiple").astype(int)
    df["is_remote_location"] = loc.str.contains("remote").astype(int)
    df["is_major_city"] = loc.apply(lambda x: int(any(c in x for c in major_cities)))
    def loc_cat(x):
        if 'multiple' in x: return 'Multiple'
        if 'remote' in x: return 'Remote'
        if any(c in x for c in major_cities): return 'Major'
        return 'Other'
    df["location_category"] = loc.apply(loc_cat)

    # --- Company features ---
    premium = ['google','microsoft','amazon','ibm']
    established = ['infosys','tcs','wipro','hcl','accenture','deloitte']
    comp = df.get("Company Name", pd.Series([""] * len(df))).str.lower().fillna("")
    df["is_premium_company"] = comp.apply(lambda x: int(any(p in x for p in premium)))
    df["is_established_company"] = comp.apply(lambda x: int(any(e in x for e in established)))
    def comp_tier(x):
        if any(p in x for p in premium): return 'Premium'
        if any(e in x for e in established): return 'Established'
        if x.strip() in ['private','confidential','na','n/a','']: return 'Private'
        return 'Other'
    df["company_tier"] = comp.apply(comp_tier)

    # --- Portal features ---
    portal = df.get("Job Portal", pd.Series([""] * len(df))).str.lower().fillna("")
    df["is_popular_portal"] = portal.apply(lambda x: int(any(p in x for p in ['linkedin','naukri','indeed'])))
    def portal_cat(x):
        if 'linkedin' in x: return 'LinkedIn'
        if 'naukri' in x: return 'Naukri'
        if 'indeed' in x: return 'Indeed'
        return 'Other'
    df["portal_category"] = portal.apply(portal_cat)

    # --- Composite features ---
    df["Number of Applicants"] = pd.to_numeric(df.get("Number of Applicants"), errors="coerce").fillna(0)
    df["competition_index"] = df["Number of Applicants"] / (df["days_to_deadline"].clip(lower=0).fillna(0) + 1)
    df["salary_exp_ratio"] = df["avg_salary"].fillna(0) / (df["avg_experience"].fillna(0) + 1)
    df["skill_demand_score"] = df.get("technical_skills_count", 0) * df.get("education_level", 0)
    df["job_attractiveness"] = (
        df["avg_salary"].fillna(0)*0.3 + df["is_popular_portal"]*10 + df["is_premium_company"]*20 +
        (1/(df["competition_index"].fillna(0)+1))*30 + df["has_remote"]*15
    )

    # --- Categorical encoding ---
    cat_cols = ['Job Type','Remote/Onsite','Company Size','location_category','company_tier','portal_category']
    label_encoders = {}
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[f"{c}_enc"] = le.fit_transform(df[c].fillna('Unknown').astype(str))
            label_encoders[c] = le

    # --- Numerical scaling ---
    num_features = [
        'min_salary','max_salary','avg_salary','min_experience','max_experience','avg_experience',
        'total_skills','technical_skills_count','desc_wordcount','education_level','competition_index',
        'salary_exp_ratio','skill_demand_score','job_attractiveness'
    ]
    num_features = [c for c in num_features if c in df.columns]
    scaler = StandardScaler()
    if num_features:
        df[[f"{c}_scaled" for c in num_features]] = scaler.fit_transform(df[num_features].fillna(0))

    # --- TF-IDF (on cleaned descriptions) ---
    LOG.info("Fitting TF-IDF (min_df=%d, max_features=%d, ngram=%d)", args.min_df, args.max_features, args.ngram)
    tfidf = TfidfVectorizer(
        ngram_range=(1, min(2, args.ngram)),
        max_features=args.max_features,
        min_df=max(1, args.min_df),
        strip_accents='unicode'
    )
    corpus = df["desc_clean"].fillna("")
    X_tfidf = tfidf.fit_transform(corpus)
    LOG.info("TF-IDF built. Vocab size: %d, matrix shape: %s", len(tfidf.vocabulary_), X_tfidf.shape)

    # --- Save outputs (meta + tfidf + preprocessors) ---
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    # choose columns to keep for meta CSV (Job ID, fake_label, numeric/encoded/scaled, a few raw counts)
    keep = [
        "Job ID", "fake_label", "Number of Applicants", "desc_wordcount", "competition_index", "job_attractiveness",
        "technical_skills_count", "total_skills",
        # India-specific and FTI features
        "india_scam_keyword_count", "has_india_scam_kw", "mentions_whatsapp",
        "mentions_gmail_or_free_email", "mentions_confidential", "has_gst_hint",
        "fraud_trigger_index"
    ]
    # also include scaled & encoded columns
    keep += [c for c in df.columns if c.endswith("_scaled") or c.endswith("_enc")]
    keep = [c for c in keep if c in df.columns]

    df[keep].to_csv(args.out_meta, index=False)
    sparse.save_npz(args.out_tfidf, X_tfidf)

    # --- Save feature group mapping (for ablation experiments) ---
    feature_groups = {
        "meta": ["Number of Applicants", "competition_index", "job_attractiveness", "education_level"],
        "salary_experience": ["min_salary", "max_salary", "avg_salary", "min_experience", "max_experience", "salary_exp_ratio"],
        "skills": ["total_skills", "technical_skills_count"] + [c for c in df.columns if c.startswith("has_") and c in df.columns],
        "india_specific": ["india_scam_keyword_count", "has_india_scam_kw", "mentions_whatsapp",
                           "mentions_gmail_or_free_email", "mentions_confidential", "has_gst_hint", "fraud_trigger_index"],
        "text": ["desc_wordcount"]  # TF-IDF handled separately
    }
    joblib.dump(feature_groups, os.path.join(args.preproc_dir, "feature_groups.joblib"))
    LOG.info("Saved feature_groups -> %s", os.path.join(args.preproc_dir, "feature_groups.joblib"))

    # save preprocessors
    joblib.dump(scaler, os.path.join(args.preproc_dir, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(args.preproc_dir, "label_encoders.joblib"))
    joblib.dump(tfidf, os.path.join(args.preproc_dir, "tfidf_vectorizer.joblib"))

    LOG.info("Saved meta features -> %s", args.out_meta)
    LOG.info("Saved TF-IDF -> %s", args.out_tfidf)
    LOG.info("Saved preprocessors -> %s", args.preproc_dir)


if __name__ == "__main__":
    main()
