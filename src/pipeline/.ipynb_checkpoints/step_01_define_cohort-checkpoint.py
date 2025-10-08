# src/pipeline/step_01_define_cohort.py

import pandas as pd
import numpy as np
import yaml
import glob
from pathlib import Path

def define_cohort(config_path: str):
    """
    Defines the study cohort by either discovering cases from a registry
    or loading a predefined set of cases, then matching controls.
    """
    # --- 1. Load Configuration ---
    print("Step 1: Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    STUDY_PARAMS = config['study_params']
    cancer_type = STUDY_PARAMS['cancer_type']
    PATHS = {key: val.format(cancer_type=cancer_type) for key, val in config['paths'].items()}
    OUTPUTS = {key: val.format(cancer_type=cancer_type) for key, val in config['outputs'].items()}
    
    Path(OUTPUTS['output_dir']).mkdir(parents=True, exist_ok=True)

    # --- 2. Handle Master Subject Log ---
    print("Step 2: Loading master subject log...")
    master_log_path = Path(PATHS['master_subject_log'])
    try:
        master_log = pd.read_csv(master_log_path)
    except FileNotFoundError:
        master_log = pd.DataFrame({'subject_id': []})
        master_log.to_csv(master_log_path, index=False)

    # --- 3. Load All Available Patients ---
    print("Step 3: Loading all available patients...")
    patient_files = glob.glob(f"{PATHS['raw_patient_data_dir']}/*.txt")
    all_patients = pd.concat(
        [pd.read_csv(f, sep='\t', usecols=['e_patid', 'e_pracid', 'gender', 'yob']) for f in patient_files],
        ignore_index=True
    ).rename(columns={'e_patid': 'subject_id'}).drop_duplicates(subset=['subject_id'])
    all_patients['subject_id'] = all_patients['subject_id'].astype('int64')
    
    available_patients = all_patients[~all_patients['subject_id'].isin(master_log['subject_id'])]

    # --- 4. Identify Cases (Based on Config Mode) ---
    mode = STUDY_PARAMS.get('cohort_definition_mode', 'discovery')
    print(f"Step 4: Identifying cases using '{mode}' mode...")

    if mode == 'predefined':
        cases_predefined = pd.read_csv(PATHS['predefined_cases_file'])
        
        # Filter for the specific cancer and ensure it's in our available patient pool
        cases_predefined = cases_predefined[cases_predefined[STUDY_PARAMS['cancer_type']] == 1].rename(columns={'epatid': 'subject_id'})
        cases_predefined['subject_id'] = cases_predefined['subject_id'].astype('int64')
        
        # Join with available_patients. This creates 'gender_x' (from this file) and 'gender_y' (from patient files)
        cases_predefined = pd.merge(cases_predefined, available_patients, on='subject_id', how='inner')

        # Derive year of birth (yob)
        # Add the specific format for dates like '20feb2014' to remove the warning
        cases_predefined['cancerdate_dt'] = pd.to_datetime(cases_predefined['cancerdate'], format='%d%b%Y')
        cases_predefined['yob'] = (cases_predefined['cancerdate_dt'].dt.year - cases_predefined['ageatindex']).astype(int)
        
        # --- FIX: Use the correct column 'gender_x' and assign to a new 'gender' column ---
        cases_predefined['gender'] = np.where(cases_predefined['gender_x'] == 'male', 1, 2)
        
        # Create the final 'cases' DataFrame with the required columns for matching
        cases = cases_predefined[['subject_id', 'e_pracid', 'gender', 'yob']].copy()


    else: # 'discovery' mode (your previous logic)
        cancer_df = pd.read_stata(PATHS['raw_cancer_data'])
        cases_with_site = cancer_df[
            (cancer_df['site'] == STUDY_PARAMS['cancer_type']) &
            (cancer_df['cancerdate'] >= pd.to_datetime(STUDY_PARAMS['start_date']))
        ].rename(columns={'epatid': 'subject_id'})
        cases_with_site['subject_id'] = cases_with_site['subject_id'].astype('int64')
        cases = pd.merge(cases_with_site[['subject_id']], available_patients, on='subject_id', how='inner')

    print(f"Found {len(cases)} valid cases for this study.")

    # --- 5. Identify Potential Controls (Same for both modes) ---
    print("Step 5: Identifying potential controls...")
    all_cancer_ids = pd.read_stata(PATHS['raw_cancer_data'])['epatid'].astype('int64').unique()
    potential_controls = available_patients[~available_patients['subject_id'].isin(all_cancer_ids)].copy()
    potential_controls = potential_controls.rename(columns={'subject_id': 'control_id', 'yob': 'control_yob'})
    print(f"Found {len(potential_controls)} potential controls.")

    # --- 6, 7, 8, 9. Match, Sample, and Save (Same for both modes) ---
    # (This section is unchanged from your uploaded script)
    print("Step 6: Matching cases to controls...")
    matches = pd.merge(cases, potential_controls, on=['e_pracid', 'gender'], how='inner')
    yob_window = STUDY_PARAMS['yob_window']
    matches = matches[matches['control_yob'].between(matches['yob'] - yob_window, matches['yob'] + yob_window)]

    print(f"Step 7: Sampling up to {STUDY_PARAMS['controls_per_case']} controls per case...")
    sampled_controls = matches.groupby('subject_id')['control_id'].apply(
        lambda x: x.sample(n=min(len(x), STUDY_PARAMS['controls_per_case']), random_state=42)
    ).reset_index(name='control_id')[['subject_id', 'control_id']]

    print("Step 8: Generating final cohort file...")
    cases_final = cases[['subject_id']].copy()
    cases_final['is_case'] = 1
    controls_final = sampled_controls[['control_id']].copy().rename(columns={'control_id': 'subject_id'}).drop_duplicates()
    controls_final['is_case'] = 0
    cohort = pd.concat([cases_final, controls_final], ignore_index=True)
    cohort.to_csv(OUTPUTS['cohort_file'], index=False)
    
    print(f"Generated cohort file with {len(cohort)} total subjects.")

    print("Step 9: Updating master subject log...")
    new_subjects_to_log = cohort[['subject_id']]
    updated_log = pd.concat([master_log, new_subjects_to_log]).drop_duplicates()
    updated_log.to_csv(master_log_path, index=False)
    
    print(f"Master log updated. Total subjects tracked: {len(updated_log)}.")
    print("--- Stage 1: Cohort Definition COMPLETE ✅ ---")


if __name__ == '__main__':
    define_cohort('config.yaml')