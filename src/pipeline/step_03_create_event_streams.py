# src/pipeline/step_03_create_event_streams.py

import polars as pl
import yaml
import glob
import os
from pathlib import Path
from src.utils.mapping_setup import setup_lookup_tables
from src.utils.drug_episodes import create_drug_episodes


# =============================================================================
# Main Pipeline Function for Stage 3
# =============================================================================

def create_event_streams(config_path: str):
    """
    Processes raw medical records for the cohort and generates final event streams.
    """
    # --- 1. Load Configuration & Subject Data ---
    print("Step 1: Loading configuration and subject information...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    OUTPUTS = config['outputs']
    
    subjects_df = pl.read_csv(OUTPUTS['subject_information_file']).with_columns(
        pl.col("cancerdate").str.to_datetime(strict=False).dt.date()
    )
    
    # Pre-load all our mapping tables
    mapping_tables = setup_lookup_tables(config)

    # # --- 2. Create Static BIRTH Event ---
    # print("Step 2: Creating static BIRTH events...")
    # birth_events = subjects_df.select(
    #     pl.col("subject_id"),
    #     pl.col("split"),
    #     time=pl.date(pl.col("yob"), 1, 1),
    #     code=pl.lit("MEDS_BIRTH"),
    #     numeric_value=pl.lit(None, dtype=pl.Float64)
    # )

    # --- 3. Scan and Standardize Raw Medical Events ---
    print("Step 3: Lazily scanning and standardizing raw events...")
    # Scan observations
    obs_files = glob.glob(f"{PATHS['observation_data_dir']}/*.txt")
    obs_scans = []
    for f in obs_files:
        try:
            file_columns = pl.scan_csv(f, separator='\t', n_rows=0).columns
            
            # Determine which patient ID column to use and create an expression
            if 'e_patid' in file_columns:
                patient_id_expr = pl.col('e_patid').alias('subject_id')
            elif 'consid' in file_columns:
                patient_id_expr = pl.col('consid').alias('subject_id')
            elif 'e_pracid' in file_columns:
                patient_id_expr = pl.col('e_pracid').alias('subject_id')
            else:
                continue # Skip file if no recognizable ID column is found

            scan = pl.scan_csv(f, separator='\t', dtypes={'medcodeid': pl.Utf8}) \
                .select(
                    patient_id_expr,
                    pl.col('obsdate').str.to_date(strict=False).alias('time'),
                    pl.lit("medcodeid//").add(pl.col("medcodeid")).alias("raw_code_full"),
                    pl.col('value').cast(pl.Float64, strict=False).alias('numeric_value'),
                    pl.lit(None, dtype=pl.Int64).alias('duration')
                )
            obs_scans.append(scan)
        except Exception as e:
            print(f"Warning: Could not process observation file {os.path.basename(f)}. Error: {e}")
    obs_lf = pl.concat(obs_scans, how="vertical")

    # Scan medications
    med_files = glob.glob(f"{PATHS['medication_data_dir']}/*drugissue*.txt")
    med_scans = []
    for f in med_files:
        try:
            file_columns = pl.scan_csv(f, separator='\t', n_rows=0).columns
            
            # Standardize patient ID column
            if 'e_patid' in file_columns:
                patient_id_expr = pl.col('e_patid').alias('subject_id')
            else:
                continue

            scan = pl.scan_csv(f, separator='\t', dtypes={'prodcodeid': pl.Utf8}) \
                .select(
                    patient_id_expr,
                    pl.col('issuedate').str.to_date(strict=False).alias('time'),
                    pl.lit("prodcodeid//").add(pl.col("prodcodeid")).alias("raw_code_full"),
                    pl.lit(None, dtype=pl.Float64).alias('numeric_value'),
                    pl.col('duration').cast(pl.Int64, strict=False)
                )
            med_scans.append(scan)
        except Exception as e:
            print(f"Warning: Could not process medication file {os.path.basename(f)}. Error: {e}")
    meds_lf = pl.concat(med_scans, how="vertical")

    # Combine all raw events into a single lazyframe
    all_events_lf = pl.concat([obs_lf, meds_lf], how="vertical") \
        .drop_nulls(subset=['time'])

    # --- 4. Filter, Map, and Process Events ---
    print("Step 4: Filtering events for cohort and mapping codes...")
    # Join with subjects to keep only cohort members and filter events after cancer diagnosis
    processed_events_lf = all_events_lf.join(subjects_df.lazy(), on='subject_id', how='inner') \
        .filter((pl.col("time") <= pl.col("cancerdate")) | pl.col("cancerdate").is_null()) \
        .select(
            pl.col('subject_id'), pl.col('split'), pl.col('time'),
            pl.col('raw_code_full'), pl.col('numeric_value'), pl.col('duration')
        )

    # Extract the raw code (e.g., the number part) for mapping
    processed_events_lf = processed_events_lf.with_columns(
        raw_code = pl.col('raw_code_full').str.extract(r"//(.*)")
    )
    
    # Apply all mappings via joins. Use coalesce to pick the first available mapping.
    mapped_lf = processed_events_lf \
        .join(mapping_tables['medcodes_map'].lazy(), on='raw_code', how='left') \
        .join(mapping_tables['read_map'].lazy(), on='raw_code', how='left') \
        .rename({'mapped_code': 'mapped_medcode', 'mapped_code_right': 'mapped_read'}) \
        .join(mapping_tables['prodcodes_map'].lazy(), on='raw_code', how='left') \
        .join(mapping_tables['dmd_map'].lazy(), on='raw_code', how='left') \
        .rename({'mapped_code': 'mapped_prodcode', 'mapped_code_right': 'mapped_dmd'}) \
        .with_columns(
            mapped_code=pl.coalesce(
                pl.col('mapped_medcode'), pl.col('mapped_read'),
                pl.col('mapped_prodcode'), pl.col('mapped_dmd')
            )
        ) \
        .filter(pl.col('mapped_code').is_not_null()) # Keep only events that were successfully mapped

    # --- 4. Process and Write Data in Batches ---
    print("Step 4: Processing and writing data in memory-efficient batches...")
    
    split_dir_map = {'train': 'train', 'val': 'tuning', 'test': 'held_out'}
    output_base_dir = Path(OUTPUTS['event_stream_dir'])
    
    for split_name, output_dir_name in split_dir_map.items():
        split_output_dir = output_base_dir / output_dir_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  - Processing split: '{split_name}' -> saving to '{output_dir_name}'")
        
        # Get patient IDs for the current split to create chunks
        patient_ids = subjects_df.filter(pl.col('split') == split_name)['subject_id'].to_list()
        
        if not patient_ids:
            continue
            
        shard_size = 1000
        for i, start_index in enumerate(range(0, len(patient_ids), shard_size)):
            end_index = start_index + shard_size
            patient_id_chunk = patient_ids[start_index:end_index]
            
            print(f"    - Processing shard {i} ({len(patient_id_chunk)} patients)...")
            
            # --- Perform the .collect() on just this chunk ---
            chunk_df = mapped_lf.filter(pl.col('subject_id').is_in(patient_id_chunk)).collect()
            
            # Perform eager operations (drug episodes, birth events) on the small chunk
            subjects_in_chunk = subjects_df.filter(pl.col('subject_id').is_in(patient_id_chunk))
            
            birth_events = subjects_in_chunk.select("subject_id", "split", time=pl.date(pl.col("yob"), 1, 1), code=pl.lit("MEDS_BIRTH"), numeric_value=pl.lit(None, dtype=pl.Float64))
            
            prescriptions_df = chunk_df.filter(pl.col('mapped_code').str.starts_with("PRESCRIPTION//"))
            non_prescription_df = chunk_df.filter(~pl.col('mapped_code').str.starts_with("PRESCRIPTION//"))
            drug_episodes_df = create_drug_episodes(prescriptions_df)
            
            # Combine all events for this chunk
            final_chunk_df = pl.concat([
                birth_events,
                non_prescription_df.select(['subject_id', 'split', 'time', pl.col('mapped_code').alias('code'), 'numeric_value']),
                drug_episodes_df.select(['subject_id', 'split', 'time', 'code']).with_columns(pl.lit(None, dtype=pl.Float64).alias('numeric_value'))
            ])
            
            # Sort and write the shard to a file
            final_chunk_df = final_chunk_df.with_columns(
                sort_priority=pl.when(pl.col('code') == "MEDS_BIRTH").then(0).otherwise(1)
            ).sort(['subject_id', 'sort_priority', 'time']).drop('sort_priority')

            output_path = split_output_dir / f"shard_{i}.parquet"
            print(f"      - Writing shard to {output_path}")
            final_chunk_df.select(['subject_id', 'time', 'code', 'numeric_value']).write_parquet(output_path)
    
    # # --- 5. Consolidate Drug Episodes and Combine All Events ---
    # print("Step 5: Consolidating prescriptions and combining all event types...")
    # # Now we collect the data to perform the episode consolidation
    # main_df = mapped_lf.collect()
    # print('Finished collection of main dataframe')
    
    # prescriptions_df = main_df.filter(pl.col('mapped_code').str.starts_with("PRESCRIPTION//"))
    # non_prescription_df = main_df.filter(~pl.col('mapped_code').str.starts_with("PRESCRIPTION//"))

    # drug_episodes_df = create_drug_episodes(prescriptions_df)
    
    # # Combine all event types: birth, non-prescriptions, and drug episodes
    # final_df = pl.concat([
    #     birth_events,
    #     non_prescription_df.select(['subject_id', 'split', 'time', pl.col('mapped_code').alias('code'), 'numeric_value']),
    #     drug_episodes_df.select(['subject_id', 'split', 'time', 'code']).with_columns(pl.lit(None, dtype=pl.Float64).alias('numeric_value'))
    # ])

    # # --- 6. Final Sort and Write to Parquet Files ---
    # print("Step 6: Sorting and writing patient event files...")
    # # Final sort: by patient, then put BIRTH first, then by time
    # final_df = final_df.with_columns(
    #     sort_priority=pl.when(pl.col('code') == "MEDS_BIRTH").then(0).otherwise(1)
    # ).sort(['subject_id', 'sort_priority', 'time']).drop('sort_priority')

    # # Create output directories
    # output_base_dir = Path(OUTPUTS['event_stream_dir'])

    # split_dir_map = {
    #     'train': 'train',
    #     'val': 'tuning',
    #     'test': 'held_out'
    # }
    
    # # Group by patient and write each 1000 patient's event stream to a separate file
    # for split_name, output_dir_name in split_dir_map.items():
    #     split_output_dir = output_base_dir / output_dir_name
    #     split_output_dir.mkdir(parents=True, exist_ok=True)
        
    #     print(f"  - Processing split: '{split_name}' -> saving to '{output_dir_name}'")
        
    #     split_df = final_df.filter(pl.col('split') == split_name)
    #     if split_df.is_empty():
    #         continue
            
    #     # Get unique patient IDs for this split to create chunks
    #     patient_ids = split_df['subject_id'].unique().to_list()
        
    #     # Define shard size
    #     shard_size = 1000
        
    #     for i, start_index in enumerate(range(0, len(patient_ids), shard_size)):
    #         end_index = start_index + shard_size
    #         patient_id_chunk = patient_ids[start_index:end_index]
            
    #         # Filter the dataframe for the current chunk of patients
    #         shard_df = split_df.filter(pl.col('subject_id').is_in(patient_id_chunk))
            
    #         output_path = split_output_dir / f"shard_{i}.parquet"
    #         print(f"    - Writing {len(patient_id_chunk)} patients to {output_path}")
    #         shard_df.select(['subject_id', 'time', 'code', 'numeric_value']).write_parquet(output_path)
    
    print(f"\nAll event stream files written to: {output_base_dir}")
    print("-" * 50)
    print("Stage 3: Event Stream Generation COMPLETE ✅")
    print("-" * 50)


if __name__ == '__main__':
    create_event_streams('config.yaml')