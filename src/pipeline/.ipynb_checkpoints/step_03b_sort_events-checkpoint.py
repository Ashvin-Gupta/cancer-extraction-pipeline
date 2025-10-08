# src/pipeline/step_03b_sort_events.py

import polars as pl
import yaml
from pathlib import Path
import time

def sort_events(config_path: str):
    """
    Stage 3b: Adds BIRTH events and performs an out-of-core sort on all events.
    
    Sorting order for each subject:
    1. The BIRTH event.
    2. Any events with a null timestamp.
    3. All remaining events in chronological order.
    """
    print("--- Running Stage 3b: Add Birth Events & Sort ---")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    STUDY_PARAMS = config['study_params']
    cancer_type = STUDY_PARAMS['cancer_type']
    PATHS = {key: val.format(cancer_type=cancer_type) for key, val in config['paths'].items()}
    OUTPUTS = {key: val.format(cancer_type=cancer_type) for key, val in config['outputs'].items()}

    # --- 1. Load Data Sources ---
    print("Step 1: Loading unsorted events and subject information...")
    # Scan the unsorted medical events from Stage 3a
    unsorted_events_lf = pl.scan_parquet(f"{OUTPUTS['intermediate_unsorted_dir']}/*.parquet") \
                           .rename({"e_patid": "subject_id"}) \
                           .select([    
                              "subject_id", 
                              "time", 
                              "code", 
                              "numeric_value", 
                              "numunitid"
                           ])

    # Scan the subject info file to get the year of birth (yob)
    subjects_lf = pl.scan_csv(OUTPUTS['subject_information_file'])

    # --- 2. Create MEDS_BIRTH Events ---
    print("Step 2: Creating MEDS_BIRTH events...")
    birth_events_lf = subjects_lf.select(
        pl.col("subject_id"),
        pl.col("yob")
    ).with_columns(
        time=pl.date(pl.col("yob"), 1, 1),
        code=pl.lit("MEDS_BIRTH"),
        numeric_value=pl.lit(None, dtype=pl.Float64),
        numunitid=pl.lit(None, dtype=pl.Int64)
    ).drop("yob")

    # --- 3. Combine All Events ---
    print("Step 3: Combining birth events and medical events...")
    all_events_lf = pl.concat([
        unsorted_events_lf,
        birth_events_lf
    ])
    
    # --- 4. Define the Custom Multi-Level Sort Key ---
    print("Step 4: Defining custom sort priority...")
    # Priority 0: The MEDS_BIRTH event
    # Priority 1: Any event where the timestamp is null
    # Priority 2: All other events
    custom_sort_key = (
        pl.when(pl.col("code") == "MEDS_BIRTH").then(0)
        .when(pl.col("time").is_null()).then(1)
        .otherwise(2)
        .alias("_sort_priority")
    )
    
    # --- 5. Apply the Sort ---
    print("Step 5: Sorting all events...")
    sorted_lf = all_events_lf.with_columns(custom_sort_key) \
        .sort("subject_id", "_sort_priority", "time") \
        .drop("_sort_priority")

    # --- 6. Save the Sorted Output ---
    output_dir = Path(OUTPUTS['intermediate_sorted_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing sorted intermediate file to: {output_dir}")
    start_time = time.time()
    sorted_lf.sink_parquet(
        pl.PartitionMaxSize(
            output_dir,
            max_size=100_000,
        )
    )
    end_time = time.time()

    print("--- Stage 3b COMPLETE ---")

if __name__ == '__main__':
    sort_events('config.yaml')