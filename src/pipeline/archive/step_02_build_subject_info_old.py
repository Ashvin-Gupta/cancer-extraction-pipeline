# src/pipeline/step_02_build_subject_info.py

import polars as pl
import pandas as pd
import yaml
import glob
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def build_subject_info(config_path: str):
    """
    Enriches the cohort with demographic, clinical, and split information using a hybrid approach.
    """
    # --- 1. Load Configuration & Initial Data ---
    print("Step 1: Loading configuration and initial cohort data...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    OUTPUTS = config['outputs']
    cohort_lf = pl.scan_csv(OUTPUTS['cohort_file'])

    # --- 2. Join Core Demographics and Cancer Info ---
    print("Step 2: Joining core demographic and cancer information...")
    patient_info_pd = pd.read_stata(PATHS['clean_ages_sex'])
    patient_info_lf = pl.from_pandas(patient_info_pd).lazy().select(pl.col('epatid').cast(pl.Int64).alias('subject_id'), pl.col('e_pracid').cast(pl.Int64), pl.col('gender'), pl.col('dobdate').dt.year().alias('yob'))
    cancer_info_pd = pd.read_stata(PATHS['raw_cancer_data'])
    cancer_info_lf = pl.from_pandas(cancer_info_pd).lazy().select(pl.col('epatid').cast(pl.Int64).alias('subject_id'), pl.col('cancerdate'), pl.col('site'))
    main_lf = cohort_lf.join(patient_info_lf, on='subject_id', how='left').join(cancer_info_lf, on='subject_id', how='left')

    # --- 3. Add Region Information ---
    print("Step 3: Adding practice region information...")
    practice_files = glob.glob(f"{PATHS['practice_data_dir']}/*.txt")
    region_lookup_lf = pl.concat([pl.scan_csv(f, separator='\t') for f in practice_files],how="vertical").select(pl.col('e_pracid').cast(pl.Int64, strict=False),pl.col('region')).drop_nulls(subset=['e_pracid']).unique(subset=['e_pracid'], keep='first')
    main_lf = main_lf.join(region_lookup_lf, on='e_pracid', how='left')

    # --- 4. Ethnicity Extraction ---
    print("Step 4a: Getting primary ethnicity from HES data...")
    hes_lf = pl.scan_csv(PATHS['hes_patient_data'], separator='\t').select(pl.col('e_patid').alias('subject_id'), pl.col('gen_ethnicity').alias('primary_ethnicity'))
    main_lf = main_lf.join(hes_lf, on='subject_id', how='left')
    
    main_df = main_lf.collect()

    print("Step 4b: Preparing fallback for subjects with missing ethnicity...")
    subjects_for_fallback = main_df.filter(
        pl.col('primary_ethnicity').is_null() | (pl.col('primary_ethnicity') == 'Unknown')
    )['subject_id']

    fallback_ethnicity_df = pl.DataFrame()
    if not subjects_for_fallback.is_empty():
        print(f"Found {len(subjects_for_fallback)} subjects needing fallback ethnicity search.")
        
        ethnicity_codelist_pd = pd.read_stata(PATHS['ethnicity_codelist'])
        ethnicity_codelist_lf = pl.from_pandas(ethnicity_codelist_pd).lazy().select(pl.col('medcodeid').cast(pl.Utf8), pl.col('ethnicity'))
        ethnicity_codes = ethnicity_codelist_lf.unique('medcodeid').collect()['medcodeid']

        # --- OPTIMISATION: Scan only the required columns from observation files ---
        obs_files = glob.glob(f"{PATHS['observation_data_dir']}/*.txt")
        obs_scans = []
        for f in obs_files:
            try:
                # 1. Peek at the file's header to find available columns
                file_columns = pl.scan_csv(f, separator='\t', n_rows=0).columns
                
                # 2. Build a list of expressions to select and rename columns
                select_expressions = []
                
                # Find the patient ID column and create an expression to standardize its name
                if 'e_patid' in file_columns:
                    select_expressions.append(pl.col('e_patid'))
                elif 'e_pracid' in file_columns:
                    select_expressions.append(pl.col('e_pracid').alias('e_patid'))
                elif 'consid' in file_columns:
                    select_expressions.append(pl.col('consid').alias('e_patid'))
                else:
                    continue # Skip file if no patient ID is found

                # Add expressions for the other essential columns
                if 'obsdate' in file_columns: select_expressions.append(pl.col('obsdate'))
                if 'medcodeid' in file_columns: select_expressions.append(pl.col('medcodeid'))
                
                scan = pl.scan_csv(f, separator='\t', dtypes={'medcodeid': pl.Utf8}) \
                         .select(select_expressions)
                obs_scans.append(scan)
            except Exception as e:
                print(f"Warning: Could not process file {f}. Error: {e}")

        fallback_obs_lf = pl.concat(obs_scans, how="vertical") \
            .filter(pl.col('e_patid').is_in(subjects_for_fallback) & pl.col('medcodeid').is_in(ethnicity_codes)) \
            .sort('obsdate')

        fallback_ethnicity_df = fallback_obs_lf.join(ethnicity_codelist_lf, on='medcodeid', how='left') \
            .group_by('e_patid').agg(pl.col('ethnicity').first().alias('fallback_ethnicity')) \
            .rename({'e_patid': 'subject_id'}) \
            .collect()

    if not fallback_ethnicity_df.is_empty():
        main_df = main_df.join(fallback_ethnicity_df, on='subject_id', how='left')
    else:
        main_df = main_df.with_columns(fallback_ethnicity=pl.lit(None, dtype=pl.Utf8))
        
    print("Step 4c: Combining primary and fallback ethnicity...")
    
    main_df = main_df.with_columns(
        pl.when(pl.col('primary_ethnicity').is_not_null() & (pl.col('primary_ethnicity') != 'Unknown'))
          .then(pl.col('primary_ethnicity'))
          .otherwise(pl.col('fallback_ethnicity'))
          .fill_null("Unknown")  
          .alias("ethnicity")
    )
    
    # --- 5. Create Train/Validation/Test Split ---
    print("Step 5: Creating stratified train/validation/test splits...")
    df_pandas = main_df.to_pandas()
    df_train, df_temp = train_test_split(df_pandas, test_size=0.2, random_state=42, stratify=df_pandas['is_case'])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['is_case'])
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    final_pandas_df = pd.concat([df_train, df_val, df_test])
    main_df = pl.from_pandas(final_pandas_df)

    # --- 6. Finalize and Save ---
    print("Step 6: Finalizing columns and saving the output file...")
    final_df = main_df.select(['subject_id', 'is_case', 'cancerdate', 'site', 'e_pracid', 'region', 'gender', 'yob', 'ethnicity', 'split'])
    final_df.write_csv(OUTPUTS['subject_information_file'])
    print(f"Final subject information file saved to: {OUTPUTS['subject_information_file']}")
    print(final_df.head())
    print("-" * 50)
    print("Stage 2: Subject Information Assembly COMPLETE ✅")
    print("-" * 50)


if __name__ == '__main__':
    build_subject_info('config.yaml')