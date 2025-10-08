import polars as pl
import yaml
import glob
import os
from pathlib import Path
import time

def extract_events(config_path: str):
    """
    Stage 3a: Extracts and standardises all raw observation events for the cohort.
    """
    print("--- Running Stage 3a: Extract & Standardise Events ---")
    
    DATE_FORMAT = "%d/%m/%Y"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    STUDY_PARAMS = config['study_params']
    cancer_type = STUDY_PARAMS['cancer_type']
    PATHS = {key: val.format(cancer_type=cancer_type) for key, val in config['paths'].items()}
    OUTPUTS = {key: val.format(cancer_type=cancer_type) for key, val in config['outputs'].items()}
    
    observation_files = glob.glob(os.path.join(PATHS['observation_data_dir'], "*.txt"))
    observation_dtypes = {
        "e_patid": pl.Int64,
        "obsdate": pl.String,      
        "medcodeid": pl.String,
        "value": pl.Float64,
        "numunitid": pl.Int64,
    }
    observations_df = pl.concat(
        [
            pl.scan_csv(
                f,
                separator="\t",
                has_header=True,
                schema_overrides=observation_dtypes
            ).select(observation_dtypes.keys())
            for f in observation_files
        ],
        how="vertical",
    )
    
    subjects_df = pl.scan_csv(OUTPUTS['subject_information_file']).rename({"subject_id": "e_patid"})
    
    obs_standardized_df = observations_df.with_columns(
        time=pl.col("obsdate").str.to_date(DATE_FORMAT, strict=False),
        code=pl.lit("medcodeid//") + pl.col("medcodeid"),
        numeric_value=pl.col("value"),
        # duration=pl.lit(None, dtype=pl.Float64),
    ).select(["e_patid", "time", "code", "numeric_value", "numunitid"])
    
    subject_filter_df = subjects_df.select(
        "e_patid",
        # Convert from string to datetime, then cast to just a date
        pl.col("cancerdate").str.to_datetime().cast(pl.Date)
    )
    
    filtered_medical_events_df = obs_standardized_df.join(
        subject_filter_df, on="e_patid", how="inner"
    ).filter(
        (pl.col("time") <= pl.col("cancerdate")) | (pl.col("cancerdate").is_null())
    ).drop("cancerdate")
    
    print('Starting save')
    start_time = time.time()
    filtered_medical_events_df.sink_parquet(
        pl.PartitionMaxSize(
            OUTPUTS['intermediate_unsorted_dir'],
            max_size=100_000,
        )
    )
    print(f'Finished save: {time.time() - start_time}')

if __name__ == '__main__':
    extract_events('config.yaml')
