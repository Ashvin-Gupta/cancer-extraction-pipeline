# src/pipeline/step_03c_process_events.py

import polars as pl
import yaml
from pathlib import Path
import os 
import glob
import time

from src.utils.mapping_setup import map_all_codes

def map_and_save_events(config_path: str):
    """
    Final stage: Maps codes, adds cancer events, and saves the final output.
    """
    print("--- Running Final Stage: Map & Save Events ---")
    
    # --- 1. Load Configuration & Data ---
    print("Step 1: Loading configuration and pre-sorted events...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    OUTPUTS = config['outputs']
    CANCER_TYPE = config['study_params']['cancer_type']

    sorted_events_lf = pl.scan_parquet(f"{OUTPUTS['intermediate_sorted_dir']}/*.parquet")
    subjects_lf = pl.scan_csv(OUTPUTS['subject_information_file']).select(["subject_id", "split", "cancerdate", "site"])

    # --- 2. Isolate BIRTH events and prepare other events for mapping ---
    print("Step 2: Isolating BIRTH events and preparing other events for mapping...")
    birth_events_lf = sorted_events_lf.filter(pl.col('code') == "MEDS_BIRTH")
    
    events_to_map_lf = sorted_events_lf.filter(pl.col('code') != "MEDS_BIRTH") \
        .with_columns(
            raw_code=pl.col("code").str.extract(r"//(.*)$", 1)
        )

    # --- 3. Map All Other Event Codes ---
    print("Step 3: Mapping all non-BIRTH event codes...")
    mapped_lf = map_all_codes(events_to_map_lf, config)
    
    combined_events_lf = pl.concat([
        birth_events_lf,
        mapped_lf.select(pl.all().exclude('raw_code', 'codelist_mapped', 'icd10_mapped'))
    ])

    print("Step 3: Deduplicating lifestyle events (keeping first instance of each term)...")
    
    LIFESTYLE_TERMS = ["Non-drinker", "Drinker - unspecified", "Drinker - within limits", "Drinker - excess/disorder", "current or ex-smoker", "current smoker", "ex-smoker", "nicotine or tobacco use", "non-smoker"]
    lifestyle_regex = "|".join(LIFESTYLE_TERMS)

    # Add a temporary column to identify the specific lifestyle term
    events_with_shortcode_lf = combined_events_lf.with_columns(
        _short_code=pl.col('code').str.extract(r"//(.*?//)", 1)
    )

    # Partition the data into lifestyle and other events
    lifestyle_events_lf = events_with_shortcode_lf.filter(pl.col('_short_code').str.contains(lifestyle_regex))
    other_events_lf = events_with_shortcode_lf.filter(~pl.col('_short_code').str.contains(lifestyle_regex))

    # Deduplicate the lifestyle events partition, keeping the first occurrence of each term
    deduplicated_lifestyle_lf = lifestyle_events_lf.unique(
        subset=['subject_id', '_short_code'], keep='first'
    )
    
    # Recombine the two partitions and drop the temporary column
    final_cleaned_lf = pl.concat([other_events_lf, deduplicated_lifestyle_lf]) \
        .drop('_short_code')
    

    # --- 4. Add Split Info, Collect, and Add Cancer Event ---
    print("Step 4: Adding split information and collecting events...")
    final_df = final_cleaned_lf.join(subjects_lf, on="subject_id", how="inner").collect()

    # --- 5. Add a cancer diagnosis event for all cases ---
    print("Step 5: Adding final cancer diagnosis events...")
    cancer_cases = subjects_lf.filter(pl.col('cancerdate').is_not_null()).collect()
    cancer_events = cancer_cases.select(
        # Positional arguments first
        pl.col('subject_id'),
        pl.col('split'),
        pl.col('cancerdate'),
        pl.col('site'),
        # Keyword arguments after
        time=pl.col('cancerdate').str.to_datetime().cast(pl.Date),
        code=pl.format(f"MEDICAL//{CANCER_TYPE}_cancer//"),
        numeric_value=pl.lit(None, dtype=pl.Float64)
    )
    
    # Reorder columns to match before concatenating
    cancer_events = cancer_events.select(final_df.columns)
    final_df = pl.concat([final_df, cancer_events])

    # --- 6. Perform Final Sort ---
    print("Step 6: Performing final sort...")
    sort_key = (
        pl.when(pl.col("code") == "MEDS_BIRTH").then(0)
        .when(pl.col("time").is_null()).then(1)
        .otherwise(2)
        .alias("_sort_priority")
    )
    # final_sorted_df = final_df.with_columns(sort_key).sort("subject_id", "_sort_priority", "time")

    final_sorted_df = final_df.sort("time") \
        .with_columns(
            # Create a temporary 'short_code' for deduplication
            _short_code=pl.when(pl.col('code').str.contains('//'))
                          .then(pl.col('code').str.extract(r"^(.*?//.*?//)", 1))
                          .otherwise(pl.col('code')) # Handles codes like 'BIRTH'
        ) \
        .unique(subset=['subject_id', 'time', '_short_code'], keep='first') \
        .drop('_short_code') \
        .with_columns(sort_key) \
        .sort("subject_id", "_sort_priority", "time") # The final sort for output

    # --- 7. Save Final Output Files ---
    print("Step 7: Saving final event stream files in shards...")
    output_base_dir = OUTPUTS['event_stream_dir']
    split_map = {'train': 'train', 'val': 'tuning', 'test': 'held_out'}
    SHARD_SIZE = 1000
    for subdir in split_map.values():
        os.makedirs(os.path.join(output_base_dir, subdir), exist_ok=True)

    for (split_name,), split_data in final_sorted_df.group_by('split'):
        if split_name not in split_map:
            continue
        print(f"Processing '{split_name}' split...")
        output_dir = os.path.join(output_base_dir, split_map[split_name])
        subject_ids = split_data.get_column('subject_id').unique().to_list()
        
        for i in range(0, len(subject_ids), SHARD_SIZE):
            shard_number = i // SHARD_SIZE
            subject_id_chunk = subject_ids[i : i + SHARD_SIZE]
            shard_data = split_data.filter(pl.col('subject_id').is_in(subject_id_chunk))
            data_to_write = shard_data.select(
                pl.col("subject_id"),
                pl.col("time").cast(pl.Datetime(time_unit="us")),
                pl.col("code"),
                pl.col("numeric_value").cast(pl.Float32).alias("value"),
                pl.lit(None, dtype=pl.Utf8).alias("text_value")
            )
            output_path = os.path.join(output_dir, f"shard_{shard_number}.parquet")
            print(f"  -> Saving shard {shard_number} with {len(subject_id_chunk)} subjects to {output_path}")
            data_to_write.write_parquet(output_path)

    print(f"\nFinal event stream files saved to: {output_base_dir}")
    print("-" * 50)
    print("Pipeline COMPLETE ✅")
    print("-" * 50)

if __name__ == '__main__':
    map_and_save_events('config.yaml')