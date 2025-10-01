# src/pipeline/step_03a_extract_events.py

import polars as pl
import yaml
import glob
import os
from pathlib import Path

def extract_events(config_path: str):
    """
    Stage 3a: Extracts and standardises all raw observation events for the cohort.
    """
    print("--- Running Stage 3a: Extract & Standardise Events ---")
    
    # --- Load Config and Subject Data ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    OUTPUTS = config['outputs']

    subjects_df = pl.read_csv(OUTPUTS['subject_information_file']).with_columns(
        pl.col("cancerdate").str.to_datetime(strict=False).dt.date()
    )

    # --- Create Static BIRTH Events ---
    birth_events_lf = subjects_df.lazy().select(
        pl.col("subject_id"),
        time=pl.date(pl.col("yob"), 1, 1),
        code=pl.lit("MEDS_BIRTH"),
        numeric_value=pl.lit(None, dtype=pl.Float64),
    )

    # --- Robustly Scan Observation Files ---
    obs_files = glob.glob(f"{PATHS['observation_data_dir']}/*.txt")
    obs_scans = []
    for f in obs_files:
        try:
            scan = pl.scan_csv(f, separator='\t', dtypes={'medcodeid': pl.Utf8}) \
                .select(
                    pl.coalesce(pl.col("e_patid"), pl.col("consid"), pl.col("e_pracid")).alias("subject_id"),
                    pl.col('obsdate').str.to_date(strict=False).alias('time'),
                    pl.lit("medcodeid//").add(pl.col("medcodeid")).alias("code"),
                    pl.col('value').cast(pl.Float64, strict=False).alias('numeric_value'),
                    pl.col('numunitid').cast(pl.Int64, strict=False)
                )
            obs_scans.append(scan)
        except Exception as e:
            print(f"Warning: Could not process observation file {os.path.basename(f)}. Error: {e}")
    obs_lf = pl.concat(obs_scans, how="vertical").drop_nulls('time')
    
    # --- Filter and Save ---
    filtered_events_lf = obs_lf.join(
        subjects_df.lazy().select("subject_id", "cancerdate"), on="subject_id", how="inner"
    ).filter(
        (pl.col("time") <= pl.col("cancerdate")) | pl.col("cancerdate").is_null()
    ).drop("cancerdate")
    
    final_lf = pl.concat([birth_events_lf, filtered_events_lf], how="vertical")
    
    output_dir = Path(OUTPUTS['intermediate_unsorted_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # # --- FIX: Provide a base filename for the output parquet files ---
    # output_file_path = output_dir / "data.parquet"
    
    print(f"Writing unsorted intermediate files to: {output_dir}")
    final_lf.select("subject_id", "time", "code", "numeric_value", "numunitid").sink_parquet(pl.PartitionMaxSize(
        output_dir,
        max_size=100_000,
    ))
    
    print("--- Stage 3a COMPLETE ---")

if __name__ == '__main__':
    extract_events('config.yaml')