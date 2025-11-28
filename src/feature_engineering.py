import argparse, logging, os, re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("feature_engineering")
RNG_SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="data/labelled_jobs.csv")
    p.add_argument("--out-meta", default="data/meta_features.csv")
    p.add_argument("--out-tfidf", default="data/desc_tfidf.npz")
    p.add_argument("--preproc-dir", default="preprocessors")
    p.add_argument("--min-tf", type=int, default=3, help="min df for TF-IDF")
    return p.parse_args()

def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def parse_salary(s):
    if pd.isna(s): return (np.nan, np.nan)
    s0 = str(s).lower().replace("lpa","").replace("inr","").replace("₹","").strip()
    nums = re.findall(r"\d+\.?\d*", s0)
    if len(nums)==1:
        v = float(nums[0]); return (v, v)
    if len(nums)>=2:
        return (float(nums[0]), float(nums[1]))
    return (np.nan, np.nan)

def parse_experience(exp):
    if pd.isna(exp): return (np.nan, np.nan)
    s = str(exp).lower()
    nums = re.findall(r"\d+\.?\d*", s)
    if len(nums)==1:
        v = float(nums[0]); return (v, v+2)
    if len(nums)>=2:
        return (float(nums[0]), float(nums[1]))
    return (np.nan, np.nan)

def word_count(text):
    if pd.isna(text): return 0
    return len(re.findall(r"\w+", str(text)))

def build_tfidf(corpus, min_df=3):
    tf = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=min_df)
    X = tf.fit_transform(corpus.fillna(""))
    return tf, X

def main():
    args = parse_args()
    Path(args.preproc_dir).mkdir(parents=True, exist_ok=True)

    LOG.info("Loading input: %s", args.input)
    df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_excel(args.input)
    LOG.info("Rows: %d columns: %d", df.shape[0], df.shape[1])

    # keep Job ID
    if 'Job ID' not in df.columns:
        df['Job ID'] = range(1, len(df)+1)

    # ensure label exists
    if 'fake_label' not in df.columns:
        LOG.warning("fake_label not found in input. Defaulting to 0.")
        df['fake_label'] = 0

    # DATE features
    df['Posted Date'] = df.get('Posted Date').apply(safe_to_datetime)
    df['Application Deadline'] = df.get('Application Deadline').apply(safe_to_datetime)
    df['posted_year'] = df['Posted Date'].dt.year.fillna(0).astype(int)
    df['posted_month'] = df['Posted Date'].dt.month.fillna(0).astype(int)
    df['posted_dayofweek'] = df['Posted Date'].dt.dayofweek.fillna(0).astype(int)
    df['days_to_deadline'] = (df['Application Deadline'] - df['Posted Date']).dt.days.fillna(-1).astype(int)
    df['is_urgent'] = (df['days_to_deadline'] <= 7).astype(int)
    df['is_long_term'] = (df['days_to_deadline'] > 30).astype(int)

    # SALARY
    salary_parsed = df.get('Salary Range', pd.Series([np.nan]*len(df))).map(parse_salary)
    df['min_salary'] = [x[0] for x in salary_parsed]
    df['max_salary'] = [x[1] for x in salary_parsed]
    df['avg_salary'] = (df['min_salary'].fillna(0) + df['max_salary'].fillna(0)) / 2
    def salary_cat(v):
        if pd.isna(v) or v==0: return 'Unknown'
        if v < 5: return 'Low'
        if v < 12: return 'Medium'
        if v < 20: return 'High'
        return 'Very High'
    df['salary_category'] = df['avg_salary'].apply(salary_cat)

    # EXPERIENCE
    exp_parsed = df.get('Experience Required', pd.Series([np.nan]*len(df))).map(parse_experience)
    df['min_experience'] = [x[0] for x in exp_parsed]
    df['max_experience'] = [x[1] for x in exp_parsed]
    df['avg_experience'] = (df['min_experience'].fillna(0) + df['max_experience'].fillna(0))/2
    def exp_cat(v):
        if pd.isna(v) or v==0: return 'Unknown'
        if v < 2: return 'Entry'
        if v < 5: return 'Mid'
        if v < 10: return 'Senior'
        return 'Exec'
    df['experience_category'] = df['avg_experience'].apply(exp_cat)

    # SKILLS (simple flags + counts)
    technical_skills = ['python','java','c++','sql','react','aws','machine learning','ui/ux','digital marketing','excel']
    def skills_feat(s):
        s = str(s).lower() if pd.notna(s) else ""
        out = {f'has_{k.replace(" ","_")}': (1 if k in s else 0) for k in technical_skills}
        tokens = [t.strip() for t in s.split(',') if t.strip()]
        out['total_skills'] = len(tokens)
        out['technical_skills_count'] = sum(out[f'has_{k.replace(" ","_")}'] for k in technical_skills)
        return out
    skills = df.get('Skills Required', pd.Series([""]*len(df))).map(skills_feat)
    skills_df = pd.DataFrame(list(skills))
    df = pd.concat([df.reset_index(drop=True), skills_df.reset_index(drop=True)], axis=1)

    # DESCRIPTION features (word count + flagged keywords)
    df['desc_wordcount'] = df.get('Job Description', pd.Series([""]*len(df))).map(lambda t: word_count(t))
    df['has_remote'] = df.get('Job Description', '').str.lower().str.contains('remote|work from home').astype(int)
    df['has_urgent'] = df.get('Job Description', '').str.lower().str.contains('urgent|immediate|start immediately').astype(int)
    df['has_fee'] = df.get('Job Description', '').str.lower().str.contains('registration fee|pay to apply|fee').astype(int)

    # LOCATION features
    major_cities = ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'chennai', 'pune', 'kolkata', 'ahmedabad', 'jaipur', 'noida']
    loc = df.get('Job Location', pd.Series([""]*len(df))).str.lower().fillna("")
    df['is_multiple_locations'] = loc.str.contains('multiple').astype(int)
    df['is_remote_location'] = loc.str.contains('remote').astype(int)
    df['is_major_city'] = loc.apply(lambda x: int(any(c in x for c in major_cities)))
    def loc_cat(x):
        if 'multiple' in x: return 'Multiple'
        if 'remote' in x: return 'Remote'
        if any(c in x for c in major_cities): return 'Major'
        return 'Other'
    df['location_category'] = loc.apply(loc_cat)

    # COMPANY features
    premium = ['google','microsoft','amazon','ibm']
    established = ['infosys','tcs','wipro','hcl','accenture','deloitte']
    comp = df.get('Company Name', pd.Series([""]*len(df))).str.lower().fillna("")
    df['is_premium_company'] = comp.apply(lambda x: int(any(p in x for p in premium)))
    df['is_established_company'] = comp.apply(lambda x: int(any(p in x for p in established)))
    def comp_tier(x):
        if any(p in x for p in premium): return 'Premium'
        if any(e in x for e in established): return 'Established'
        if x.strip() in ['private','confidential','na','n/a','']: return 'Private'
        return 'Other'
    df['company_tier'] = comp.apply(comp_tier)

    # PORTAL features
    portal = df.get('Job Portal', pd.Series([""]*len(df))).str.lower().fillna("")
    df['is_popular_portal'] = portal.apply(lambda x: int(any(p in x for p in ['linkedin','naukri','indeed'])))
    def portal_cat(x):
        if 'linkedin' in x: return 'LinkedIn'
        if 'naukri' in x: return 'Naukri'
        if 'indeed' in x: return 'Indeed'
        return 'Other'
    df['portal_category'] = portal.apply(portal_cat)

    # COMPOSITES
    df['competition_index'] = df['Number of Applicants'].fillna(0) / (df['days_to_deadline'].clip(lower=0).fillna(0) + 1)
    df['salary_exp_ratio'] = df['avg_salary'].fillna(0) / (df['avg_experience'].fillna(0) + 1)
    df['skill_demand_score'] = df.get('technical_skills_count',0) * df.get('education_level',0)
    df['job_attractiveness'] = (
        df['avg_salary'].fillna(0)*0.3 + df['is_popular_portal']*10 + df['is_premium_company']*20 +
        (1/(df['competition_index'].fillna(0)+1))*30 + df['has_remote']*15
    )

    # CATEGORICAL ENCODING (LabelEncoder)
    cat_cols = ['Job Type','Remote/Onsite','Company Size','salary_category','experience_category','location_category','company_tier','portal_category']
    encoders = {}
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[f'{c}_enc'] = le.fit_transform(df[c].fillna('Unknown').astype(str))
            encoders[c] = le

    # SCALE numerical features and save scaler
    num_features = ['min_salary','max_salary','avg_salary','min_experience','max_experience','avg_experience',
                    'total_skills','technical_skills_count','desc_wordcount','education_level','competition_index',
                    'salary_exp_ratio','skill_demand_score','job_attractiveness']
    num_features = [c for c in num_features if c in df.columns]
    scaler = StandardScaler()
    df[[f'{c}_scaled' for c in num_features]] = scaler.fit_transform(df[num_features].fillna(0))

    # TF-IDF for Job Description
    LOG.info("Fitting TF-IDF vectorizer on Job Description")
    tfidf_vec, X_tfidf = build_tfidf(df.get('Job Description', pd.Series([""]*len(df))), min_df=args.min_tf)
    LOG.info("TF-IDF shape: %s", X_tfidf.shape)

    # Save outputs
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    # keep Job ID and fake_label for traceability
    save_cols = ['Job ID','fake_label'] + [c for c in df.columns if c.endswith('_scaled') or c.endswith('_enc') or c in ['desc_wordcount','Number of Applicants','competition_index','job_attractiveness','skill_demand_score']]
    # ensure at least some columns
    save_cols = [c for c in save_cols if c in df_to_save.columns]
    df_to_save[save_cols + []].to_csv(args.out_meta, index=False)
    # Save tfidf
    sparse.save_npz(args.out_tfidf, X_tfidf)

    # Save preprocessors
    joblib.dump(scaler, os.path.join(args.preproc_dir, 'scaler.joblib'))
    joblib.dump(encoders, os.path.join(args.preproc_dir, 'label_encoders.joblib'))
    joblib.dump(tfidf_vec, os.path.join(args.preproc_dir, 'tfidf_vectorizer.joblib'))

    LOG.info("Saved meta features to %s", args.out_meta)
    LOG.info("Saved TF-IDF to %s and preprocessors to %s", args.out_tfidf, args.preproc_dir)

if __name__ == "__main__":
    main()
