# src/pipeline/step_03a_extract_events.py

import polars as pl
import yaml
import glob
import os
from pathlib import Path
import time

def extract_events(config_path: str):
    """
    Stage 3a: Extracts and standardises all raw observation events for the cohort,
    applying dynamic trajectory windows in a single, memory-efficient pass.
    """
    print("--- Running Stage 3a: Extract & Standardise Events (Optimized) ---")
    
    # --- 1. Load Configuration and Raw Data ---
    print("Step 1: Loading configuration and raw observation data...")
    DATE_FORMAT = "%d/%m/%Y"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    STUDY_PARAMS = config['study_params']
    cancer_type = STUDY_PARAMS['cancer_type']
    PATHS = {key: val.format(cancer_type=cancer_type) for key, val in config['paths'].items()}
    OUTPUTS = {key: val.format(cancer_type=cancer_type) for key, val in config['outputs'].items()}
    
    observation_files = glob.glob(os.path.join(PATHS['observation_data_dir'], "*.txt"))
    observation_dtypes = {
        "e_patid": pl.Int64, "obsdate": pl.String, "medcodeid": pl.String,
        "value": pl.Float64, "numunitid": pl.Int64,
    }
    # Lazily scan all observation files and standardize the time column
    obs_standardized_lf = pl.concat(
        [pl.scan_csv(f, separator="\t", has_header=True, schema_overrides=observation_dtypes).select(observation_dtypes.keys()) for f in observation_files],
        how="vertical",
    ).with_columns(
        time=pl.col("obsdate").str.to_date(DATE_FORMAT, strict=False)
    )

    # --- 2. Load Subject Data ---
    print("Step 2: Loading subject data...")
    subjects_lf = pl.scan_csv(OUTPUTS['subject_information_file']) \
        .rename({"subject_id": "e_patid"}) \
        .with_columns(
            cancerdate=pl.col("cancerdate").str.to_datetime().cast(pl.Date)
        )

    # --- 3. Build and Apply Trajectory Filter in a Single Pass ---
    print("Step 3: Calculating trajectory windows and filtering events...")
    
    # Join subject info onto the full event stream first
    events_with_context_lf = obs_standardized_lf.join(
        subjects_lf, on="e_patid", how="inner"
    )
    
    events_with_dates_lf = events_with_context_lf.with_columns(
        last_event_date=pl.max("time").over("e_patid")
    ).with_columns(
        end_date=pl.when(pl.col("is_case") == 1)
                   .then(pl.col("cancerdate"))
                   .otherwise(pl.col("last_event_date").dt.offset_by("-1y")),
        start_date=pl.when(pl.col("is_case") == 1)
                    .then(pl.col("cancerdate").dt.offset_by("-5y"))
                    .otherwise(pl.col("last_event_date").dt.offset_by("-6y"))
    )

    # Step 3b: Apply the filter in a separate step
    filtered_medical_events_lf = events_with_dates_lf.filter(
        pl.col("time").is_between(pl.col("start_date"), pl.col("end_date"))
    )
    
    # --- Optional Debugging Block ---
    # To use this, uncomment the lines and set a patient ID.
    # It will print the calculated dates for one patient before saving.
    # -----------------------------------------------------------
    # patient_to_debug = 362864450976 
    # if patient_to_debug:
    #     print(f"--- Debugging patient {patient_to_debug} ---")
    #     debug_df = events_with_dates_lf.filter(pl.col('e_patid') == patient_to_debug).collect()
    #     print(debug_df.select("e_patid", "time", "is_case", "start_date", "end_date"))
    #     print("--- End Debugging ---")
    # -----------------------------------------------------------

    # Final selection of columns
    final_lf = filtered_medical_events_lf.select(
        "e_patid", 
        "time", 
        "numunitid",
        code=pl.lit("medcodeid//") + pl.col("medcodeid"),
        numeric_value=pl.col("value")
    )

    # --- 4. Save the Filtered Events ---
    print('Step 4: Saving filtered, unsorted events...')
    output_dir = Path(OUTPUTS['intermediate_unsorted_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    final_lf.sink_parquet(
        output_dir / "data.parquet",
        compression='snappy'
    )
    print(f'Finished save in: {time.time() - start_time:.2f} seconds')
    print("--- Stage 3a COMPLETE ---")


if __name__ == '__main__':
    extract_events('config.yaml')