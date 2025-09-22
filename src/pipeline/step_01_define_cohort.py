# src/pipeline/step_01_define_cohort.py

import pandas as pd
import yaml
import glob
from pathlib import Path

def define_cohort(config_path: str):
    """
    Defines the study cohort using Pandas by identifying cancer cases and matching controls.
    """
    # --- 1. Load Configuration ---
    print("Step 1: Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    STUDY_PARAMS = config['study_params']
    PATHS = config['paths']
    OUTPUTS = config['outputs']
    
    Path(OUTPUTS['output_dir']).mkdir(parents=True, exist_ok=True)

    # --- 2. Handle Master Subject Log ---
    print("Step 2: Loading master subject log...")
    master_log_path = Path(PATHS['master_subject_log'])
    try:
        master_log = pd.read_csv(master_log_path)
        print(f"Found master log with {len(master_log)} subjects to exclude.")
    except FileNotFoundError:
        print("Master log not found. Creating a new one.")
        master_log = pd.DataFrame({'subject_id': []})
        master_log.to_csv(master_log_path, index=False)

    # --- 3. Load All Available Patients ---
    print("Step 3: Loading all available patients and excluding used subjects...")
    patient_files = glob.glob(f"{PATHS['raw_patient_data_dir']}/*.txt")
    all_patients = pd.concat(
        [pd.read_csv(f, sep='\t', usecols=['e_patid', 'e_pracid', 'gender', 'yob']) for f in patient_files],
        ignore_index=True
    ).rename(columns={'e_patid': 'subject_id'}).drop_duplicates(subset=['subject_id'])
    all_patients['subject_id'] = all_patients['subject_id'].astype('int64')
    
    available_patients = all_patients[~all_patients['subject_id'].isin(master_log['subject_id'])]

    # --- 4. Identify Cases ---
    print(f"Step 4: Identifying cases for cancer type: '{STUDY_PARAMS['cancer_type']}'...")
    cancer_df = pd.read_stata(PATHS['raw_cancer_data'])
    
    # Keep the 'site' column to use later
    cases_with_site = cancer_df[
        (cancer_df['site'] == STUDY_PARAMS['cancer_type']) &
        (cancer_df['cancerdate'] >= pd.to_datetime(STUDY_PARAMS['start_date']))
    ].rename(columns={'epatid': 'subject_id'})

    cases_with_site['subject_id'] = cases_with_site['subject_id'].astype('int64')
    
    # Join with available_patients to get demographics and ensure they are in our available pool
    cases = pd.merge(cases_with_site[['subject_id', 'site']], available_patients, on='subject_id', how='inner').drop_duplicates(subset=['subject_id'])
    print(f"Found {len(cases)} valid cases for this study.")

    # --- 5. Identify Potential Controls ---
    print("Step 5: Identifying potential controls...")
    all_cancer_ids = cancer_df['epatid'].astype('int64').unique()
    
    potential_controls = available_patients[~available_patients['subject_id'].isin(all_cancer_ids)].copy()
    potential_controls = potential_controls.rename(columns={'subject_id': 'control_id', 'yob': 'control_yob'})
    print(f"Found {len(potential_controls)} potential controls.")

    # --- 6. Match Cases to Controls ---
    print("Step 6: Matching cases to controls...")
    matches = pd.merge(cases, potential_controls, on=['e_pracid', 'gender'], how='inner')
    yob_window = STUDY_PARAMS['yob_window']
    matches = matches[matches['control_yob'].between(matches['yob'] - yob_window, matches['yob'] + yob_window)]

    # --- 7. Sample Controls for Each Case ---
    print(f"Step 7: Sampling up to {STUDY_PARAMS['controls_per_case']} controls per case...")
    sampled_controls = matches.groupby('subject_id')['control_id'].apply(
        lambda x: x.sample(n=min(len(x), STUDY_PARAMS['controls_per_case']), random_state=42)
    ).reset_index(name='control_id')[['subject_id', 'control_id']]

    # --- 8. Generate Final Cohort File ---
    print("Step 8: Generating final cohort file for verification...")
    # --- FIX: Add 'site' column to the output for easy checking ---
    cases_final = cases[['subject_id', 'site']].copy()
    cases_final['is_case'] = 1
    
    controls_final = sampled_controls[['control_id']].copy().rename(columns={'control_id': 'subject_id'}).drop_duplicates()
    controls_final['is_case'] = 0
    controls_final['site'] = 'Control' # Add a placeholder for controls
    
    cohort = pd.concat([cases_final, controls_final], ignore_index=True)
    # Reorder columns for clarity in the output CSV
    cohort = cohort[['subject_id', 'is_case', 'site']]
    
    cohort.to_csv(OUTPUTS['cohort_file'], index=False)
    
    print(f"Generated cohort file with {len(cohort)} total subjects ({len(cases_final)} cases, {len(controls_final)} controls).")
    print(f"Cohort file saved to: {OUTPUTS['cohort_file']}")

    # --- 9. Update Master Log ---
    print("Step 9: Updating master subject log...")
    # The master log only needs the subject_id
    new_subjects_to_log = cohort[['subject_id']]
    updated_log = pd.concat([master_log, new_subjects_to_log]).drop_duplicates()
    updated_log.to_csv(master_log_path, index=False)
    
    print(f"Master log updated. Total subjects tracked: {len(updated_log)}.")
    print("-" * 50)
    print("Stage 1: Cohort Definition COMPLETE ✅")
    print("-" * 50)


if __name__ == '__main__':
    define_cohort('config.yaml')