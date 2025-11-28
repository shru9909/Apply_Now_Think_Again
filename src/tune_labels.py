import pandas as pd
from pathlib import Path

IN = Path("data/labelled_jobs.csv")
OUT = Path("data/labelled_jobs_tuned.csv")
OUT_CLEAN = Path("data/cleaned_jobs_tuned.csv")

df = pd.read_csv(IN)
# ensure num_flags exists (compute if not)
if 'num_flags' not in df.columns:
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    df['num_flags'] = df[flag_cols].sum(axis=1)

def tune_label(row):
    # if 2 or more flags => fake
    if row['num_flags'] >= 2:
        return 1
    # if exactly 1 flag, and it's not missing_description, mark fake
    if row['num_flags'] == 1:
        rules = str(row.get('triggered_rules','')).split(',')
        # if the *only* triggered rule is missing_description, treat as non-fake
        rules = [r.strip() for r in rules if r.strip()]
        if len(rules)==1 and rules[0]=='missing_description':
            return 0
        return 1
    return 0

df['fake_label_tuned'] = df.apply(tune_label, axis=1).astype(int)

# Save full and cleaned
df.to_csv(OUT, index=False)
cols_to_keep = ['Job ID','Job Title','Company Name','Job Location','Salary Range','Experience Required',
                'Job Portal','Number of Applicants','Company Size','Job Description','Recruiter Email',
                'fake_label_tuned','triggered_rules','num_flags']
cols_present = [c for c in cols_to_keep if c in df.columns]
df[cols_present].to_csv(OUT_CLEAN, index=False)

print("Saved tuned labelled file:", OUT)
print("Label distribution (tuned):")
print(df['fake_label_tuned'].value_counts())