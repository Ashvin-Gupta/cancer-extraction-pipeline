import polars as pl
import yaml

def profile_measurements(config_path: str):
    """
    Scans the event stream to produce a summary of all
    measurements, their units, and their distributions. 
    Looks at each measruement and lab test and records the 
    quantile bins. 
    """
    print("--- Running Measurement Profiling Script on Final Data ---")
    
    # --- 1. Load Config and Final Mapped Data ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    OUTPUTS = config['outputs']

    print("Step 1: Scanning final sharded event stream...")
    # Scan all the sharded parquet files from your final output directory
    events_lf = pl.scan_parquet(f"{OUTPUTS['event_stream_dir']}/**/*.parquet")

    # --- 2. Isolate and Parse Measurement Events ---
    print("Step 2: Isolating and parsing measurement events...")
    measurements_lf = events_lf.filter(
        # (pl.col('code').str.starts_with("LAB//") | pl.col('code').str.starts_with("MEASUREMENT//")) &
        
        pl.col('numeric_value').is_not_null() &
        pl.col('numunitid').is_not_null()
    ).with_columns(
        # Extract the term (e.g., 'Hemoglobin' or '42W..00') from the code
        identifier=pl.col('code').str.extract(r"//(.*?//)", 1).str.replace_all("/", "")
    )

    # --- 3. Group by Identifier and Unit, then Calculate Stats ---
    print("Step 3: Grouping by measurement and calculating statistics...")
    
    # Define the quantile bins you want to calculate
    quantiles = [i / 10.0 for i in range(1, 10)] # Deciles 0.1, 0.2, ... 0.9
    
    profile_df = measurements_lf.group_by("identifier", "numunitid").agg(
        pl.count().alias("count"),
        pl.min("numeric_value").alias("min"),
        pl.max("numeric_value").alias("max"),
        pl.mean("numeric_value").alias("mean"),
        
        *[pl.quantile("numeric_value", q).alias(f"quantile_{int(q*100)}") for q in quantiles]
    ).sort("identifier", "count", descending=[False, True]).collect()

    # --- 4. Save the Profile ---
    output_path = OUTPUTS['profile_measurement']
    profile_df.write_csv(output_path)
    
    print("\n--- Profiling COMPLETE ---")
    print(f"✅ Summary of all measurements saved to: {output_path}")

if __name__ == '__main__':
    profile_measurements('config.yaml')