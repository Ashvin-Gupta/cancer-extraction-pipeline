import polars as pl
import yaml
import os
import csv
import pandas as pd

def clean_events(config_path: str):
    """
    Final cleaning stage: Applies curated rules to LAB tests and automated
    outlier detection to MEASUREMENT tests.
    """
    print("--- Running Final Stage: Clean & Standardize Events ---")
    
    # --- 1. Load Configuration, Data, and Rules ---
    print("Step 1: Loading configuration, sharded events, and cleaning rules...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    STUDY_PARAMS = config['study_params']
    cancer_type = STUDY_PARAMS['cancer_type']
    PATHS = {key: val.format(cancer_type=cancer_type) for key, val in config['paths'].items()}
    OUTPUTS = {key: val.format(cancer_type=cancer_type) for key, val in config['outputs'].items()}
    
    events_lf = pl.scan_parquet(f"{OUTPUTS['event_stream_dir']}/**/*.parquet")
    try:
        rules_df_pd = pd.read_csv(PATHS['cleaning_rules_final'])
        
        # Clean the column names (strips whitespace and \r characters)
        rules_df_pd.columns = rules_df_pd.columns.str.strip()
        
        # Convert to a Polars DataFrame to continue with the pipeline
        rules_df = pl.from_pandas(rules_df_pd)
        
        # Ensure the key columns are the correct float type after loading
        rules_df = rules_df.with_columns(
            pl.col("ConversionFactor").cast(pl.Float32),
            pl.col("ConversionBias").cast(pl.Float32),
            pl.col("ValidMin").cast(pl.Float32),
            pl.col("ValidMax").cast(pl.Float32)
        )
    except FileNotFoundError:
        print(f"Warning: Cleaning rules file not found at '{PATHS['cleaning_rules_final']}'. Skipping cleaning.")
        # In a real run, you might want to exit or handle this differently
        return

    # --- 2. Partition the Event Stream ---
    print("Step 2: Partitioning event stream into LAB, MEASUREMENT, and OTHER...")
    
    # Isolate events that need cleaning (must have a numeric value)
    events_with_value_lf = events_lf.filter(pl.col('numeric_value').is_not_null())
    other_events_lf = events_lf.filter(pl.col('numeric_value').is_null())

    lab_events_lf = events_with_value_lf.filter(pl.col('code').str.starts_with("LAB//"))
    measurement_events_lf = events_with_value_lf.filter(pl.col('code').str.starts_with("MEASUREMENT//"))
    # Any other events with values that aren't LAB or MEASUREMENT will be passed through
    medical_value_events_lf = events_with_value_lf.filter(
        ~pl.col('code').str.starts_with("LAB//") & ~pl.col('code').str.starts_with("MEASUREMENT//")
    )

    # --- 3. Process the LAB Stream (Curated Cleaning) ---
    print("Step 3: Applying curated rules to LAB tests...")
    
    # Prepare the rules lookup table
    lab_rules_df = rules_df.filter(pl.col("IdentifierType") == "MedicalTerm")

    cleaned_lab_lf = lab_events_lf.with_columns(
        Identifier=pl.col('code').str.extract(r"//(.*?//)", 1).str.replace_all("/", "")
    ).join(
        lab_rules_df.lazy(),
        left_on=["Identifier", "numunitid"],
        right_on=["Identifier", "UnitID"],
        how="left"
    ).with_columns(
        # Apply the linear transformation
        standardized_value = (pl.col('numeric_value') * pl.col('ConversionFactor')) + pl.col('ConversionBias').fill_null(0)
    ).with_columns(
        # Set values outside the valid range to null
        cleaned_value=pl.when(
            pl.col('standardized_value').is_between(pl.col('ValidMin'), pl.col('ValidMax'))
        ).then(pl.col('standardized_value')).otherwise(pl.lit(None))
    ).select(
        pl.col("subject_id"), pl.col("time"), pl.col("code"),
        pl.col("cleaned_value").alias("numeric_value"), # Replace original value
        pl.col("text_value"), pl.col("numunitid")
    )
    
    # --- 4. Process the MEASUREMENT Stream (Automated Cleaning) ---
    print("Step 4: Applying automated outlier detection to MEASUREMENT tests...")

    # First, calculate stats (median, std) for each measurement group
    stats_lf = measurement_events_lf.with_columns(
        Identifier=pl.col('code').str.extract(r"//(.*?//)", 1).str.replace_all("/", "")
    ).group_by('Identifier').agg(
        pl.median('numeric_value').alias('median'),
        pl.std('numeric_value').alias('std')
    )

    # Join stats back and filter outliers
    cleaned_measurement_lf = measurement_events_lf.with_columns(
        Identifier=pl.col('code').str.extract(r"//(.*?//)", 1).str.replace_all("/", "")
    ).join(
        stats_lf, on='Identifier', how='left'
    ).with_columns(
        lower_bound = pl.max_horizontal(0, pl.col('median') - 3 * pl.col('std')),
        upper_bound = pl.col('median') + 3 * pl.col('std')
    ).filter(
        pl.col('numeric_value').is_between(pl.col('lower_bound'), pl.col('upper_bound'))
    ).select(
        pl.col("subject_id"), pl.col("time"), pl.col("code"), pl.col("numeric_value"),
        pl.col("text_value"), pl.col("numunitid")
              
    )

    # --- 5. Recombine and Save ---
    print("Step 5: Recombining all streams and saving final output...")
    
    # Collect all parts. Using .collect() on each lazy frame before concat.
    final_df = pl.concat([
        other_events_lf.collect(),
        medical_value_events_lf.collect(),
        cleaned_lab_lf.collect(),
        cleaned_measurement_lf.collect()
    ]).sort("subject_id", "time")
    
    # Get split info for saving
    subjects_df = pl.read_csv(OUTPUTS['subject_information_file']).select(["subject_id", "split"])
    final_df = final_df.join(subjects_df, on="subject_id", how="inner")

    output_base_dir = OUTPUTS['final_cleaned_dir']
    split_map = {'train': 'train', 'val': 'tuning', 'test': 'held_out'}
    SHARD_SIZE = 1000
    for subdir in split_map.values():
        os.makedirs(os.path.join(output_base_dir, subdir), exist_ok=True)

    for (split_name,), split_data in final_df.group_by('split'):
        if split_name not in split_map: continue
        print(f"Processing '{split_name}' split...")
        output_dir = os.path.join(output_base_dir, split_map[split_name])
        subject_ids = split_data.get_column('subject_id').unique().to_list()
        
        for i in range(0, len(subject_ids), SHARD_SIZE):
            shard_number = i // SHARD_SIZE
            subject_id_chunk = subject_ids[i : i + SHARD_SIZE]
            shard_data = split_data.filter(pl.col('subject_id').is_in(subject_id_chunk))
            
            # Select final columns for output
            data_to_write = shard_data.select(
                "subject_id", "time", "code", "numeric_value", "text_value"
            )
            output_path = os.path.join(output_dir, f"shard_{shard_number}.parquet")
            print(f"  -> Saving cleaned shard {shard_number} to {output_path}")
            data_to_write.write_parquet(output_path)
    
    print(f"\nFinal cleaned event stream files saved to: {output_base_dir}")
    print("-" * 50)
    print("Cleaning and Standardization COMPLETE ✅")
    print("-" * 50)

if __name__ == '__main__':
    clean_events('config.yaml')