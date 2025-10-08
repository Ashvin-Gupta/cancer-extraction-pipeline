# src/pipeline/step_02_build_subject_info.py

import polars as pl
import pandas as pd
import yaml
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

def build_subject_info(config_path: str):
    """
    Enriches the cohort, prioritizing data from a predefined case file if provided.
    """
    # --- 1. Load Configuration & Initial Data ---
    print("Step 1: Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    STUDY_PARAMS = config['study_params']
    cancer_type = STUDY_PARAMS['cancer_type']
    PATHS = {key: val.format(cancer_type=cancer_type) for key, val in config['paths'].items()}
    OUTPUTS = {key: val.format(cancer_type=cancer_type) for key, val in config['outputs'].items()}
    
    cohort_lf = pl.scan_csv(OUTPUTS['cohort_file'])

    # --- 2. Join Core Demographics and Cancer Info ---
    # (This section is unchanged from your uploaded script)
    print("Step 2: Joining core demographic and cancer information...")
    patient_info_pd = pd.read_stata(PATHS['clean_ages_sex'])
    patient_info_lf = pl.from_pandas(patient_info_pd).lazy().select(pl.col('epatid').cast(pl.Int64).alias('subject_id'), pl.col('e_pracid').cast(pl.Int64), pl.col('gender'), pl.col('dobdate').dt.year().alias('yob'))
    cancer_info_pd = pd.read_stata(PATHS['raw_cancer_data'])
    cancer_info_lf = pl.from_pandas(cancer_info_pd).lazy().select(pl.col('epatid').cast(pl.Int64).alias('subject_id'), pl.col('cancerdate'), pl.col('site'))
    main_lf = cohort_lf.join(patient_info_lf, on='subject_id', how='left').join(cancer_info_lf, on='subject_id', how='left')

    # --- 3. Join Region & Predefined Case Info ---
    print("Step 3: Joining region and predefined case information...")
    practice_files = glob.glob(f"{PATHS['practice_data_dir']}/*.txt")
    region_lookup_lf = pl.concat([pl.scan_csv(f, separator='\t') for f in practice_files],how="vertical").select(pl.col('e_pracid').cast(pl.Int64, strict=False),pl.col('region')).drop_nulls(subset=['e_pracid']).unique(subset=['e_pracid'], keep='first')
    main_lf = main_lf.join(region_lookup_lf, on='e_pracid', how='left')

    # --- NEW: Join data from the predefined cases file ---
    if STUDY_PARAMS.get('cohort_definition_mode') == 'predefined':
        predefined_cases_lf = pl.scan_csv(PATHS['predefined_cases_file']) \
            .select(
                pl.col('epatid').alias('subject_id'),
                pl.col('ethnicity').alias('predefined_ethnicity'),
                pl.col('smokingstatus'),
                pl.col('imd')
            )
        main_lf = main_lf.join(predefined_cases_lf, on='subject_id', how='left')
    else:
        # If not in predefined mode, create empty columns to keep the schema consistent
        main_lf = main_lf.with_columns(
            predefined_ethnicity=pl.lit(None, dtype=pl.Utf8),
            smokingstatus=pl.lit(None, dtype=pl.Utf8),
            imd=pl.lit(None, dtype=pl.Int64)
        )
        
    # --- 4. Ethnicity Extraction with Fallback ---
    print("Step 4a: Getting primary ethnicity from HES data...")
    hes_lf = pl.scan_csv(PATHS['hes_patient_data'], separator='\t').select(pl.col('e_patid').alias('subject_id'), pl.col('gen_ethnicity').alias('hes_ethnicity'))
    main_lf = main_lf.join(hes_lf, on='subject_id', how='left')
    main_df = main_lf.collect()

    print("Step 4b: Preparing fallback for subjects with missing ethnicity...")
    # Identify subjects who need fallback (controls, or cases missing data)
    subjects_for_fallback = main_df.filter(
        pl.col('predefined_ethnicity').is_null() & 
        (pl.col('hes_ethnicity').is_null() | (pl.col('hes_ethnicity') == 'Unknown'))
    )['subject_id']

    fallback_ethnicity_df = pl.DataFrame()
    if not subjects_for_fallback.is_empty():
        # (The fallback logic itself is unchanged from your uploaded script)
        ethnicity_codelist_pd = pd.read_stata(PATHS['ethnicity_codelist'])
        ethnicity_codelist_lf = pl.from_pandas(ethnicity_codelist_pd).lazy().select(pl.col('medcodeid').cast(pl.Utf8), pl.col('ethnicity'))
        ethnicity_codes = ethnicity_codelist_lf.unique('medcodeid').collect()['medcodeid']
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
        
        
    print("Step 4c: Combining ethnicity sources with priority...")
    main_df = main_df.with_columns(
        ethnicity=pl.coalesce(
            pl.col('predefined_ethnicity'),
            pl.col('hes_ethnicity'),
            pl.col('fallback_ethnicity')
        ).fill_null("Unknown")
    )
    
    # --- 5. Create Train/Validation/Test Split ---
    # (This section is unchanged from your uploaded script)
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
    # --- NEW: Add 'imd' and 'smokingstatus' to the final output ---
    final_df = main_df.select(['subject_id', 'is_case', 'cancerdate', 'site', 'e_pracid', 'region', 'gender', 'yob', 'ethnicity', 'imd', 'smokingstatus', 'split'])
    final_df.write_csv(OUTPUTS['subject_information_file'])
    
    print(f"Final subject information file saved to: {OUTPUTS['subject_information_file']}")
    print("--- Stage 2: Subject Information Assembly COMPLETE ✅ ---")


if __name__ == '__main__':
    build_subject_info('config.yaml')