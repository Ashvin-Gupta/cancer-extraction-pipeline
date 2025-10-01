# analyze_mapping_coverage.py

import polars as pl
import yaml
from src.utils.mapping_setup import  expand_codes
import pandas as pd

def analyze_coverage(config_path: str):
    """
    Analyzes a sample of raw medcodes to report how many are mapped
    by each rule in the fallback hierarchy.
    """
    print("--- Analyzing Mapping Coverage ---")
    
    # --- 1. Load Config and a sample of unique raw codes ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    OUTPUTS = config['outputs']

    print("Step 1: Getting a sample of unique raw codes from your data...")
    raw_codes_to_check_lf = pl.scan_parquet(f"{OUTPUTS['intermediate_sorted_dir']}/*.parquet") \
        .filter(pl.col('code') != "MEDS_BIRTH") \
        .select(
            raw_code=pl.col("code").str.extract(r"//(.*)$", 1)
        ) \
        .drop_nulls() \
        .unique() \
        .head(100_000)
        
    df_codes_to_check = raw_codes_to_check_lf.collect()
    total_codes = len(df_codes_to_check)
    if total_codes == 0:
        print("No codes found to analyze.")
        return
    print(f"Analyzing {total_codes} unique raw codes...")

    # --- 2. Build the same mapping tables used in the pipeline ---
    print("Step 2: Building the mapping tables...")

    # --- FIX: Use the robust Pandas reader for the problematic codelist CSV ---
    try:
        codelists_pd = pd.read_csv(PATHS['cleaned_codelists'], skip_blank_lines=True)
        codelists_pd.columns = codelists_pd.columns.str.strip()
        codelists_lf = pl.from_pandas(codelists_pd).lazy()
    except Exception as e:
        print(f"FATAL: Could not read the codelist file: {e}")
        return
    
    # --- The rest of the logic uses the cleaned codelists_lf ---
    map1_df = expand_codes(codelists_lf, "medcodes", "MedicalTerm").rename({"code": "raw_code", "MedicalTerm": "map1_term"}).unique(subset=['raw_code'], keep='first').collect()
    medcode_to_readcode_df = pl.scan_csv(PATHS['medical_dictionary']).select(pl.col("MedCodeId").cast(pl.Utf8).alias("raw_code"), pl.col("OriginalReadCode").alias("read_code")).drop_nulls().unique(subset=['raw_code'], keep='first').collect()
    map3_df = expand_codes(codelists_lf, "ReadcodeList", "MedicalTerm").rename({"code": "read_code", "MedicalTerm": "map3_term"}).unique(subset=['read_code'], keep='first').collect()
    map4_df = expand_codes(codelists_lf, "medcodes2", "MedicalTerm").rename({"code": "raw_code", "MedicalTerm": "map4_term"}).unique(subset=['raw_code'], keep='first').collect()

    # --- 3. Perform sequential joins to see where codes get mapped ---
    print("Step 3: Joining mapping tables to the sample codes...")
    results_df = df_codes_to_check \
        .join(map1_df, on="raw_code", how="left") \
        .join(medcode_to_readcode_df, on="raw_code", how="left") \
        .join(map3_df, on="read_code", how="left") \
        .join(map4_df, on="raw_code", how="left")

    # --- 4. Calculate and Print the Statistics ---
    print("\n--- MAPPING COVERAGE REPORT ---")
    
    map1_count = results_df.get_column("map1_term").is_not_null().sum()
    print(f"Mapped by primary 'medcodes' list:      {map1_count:>6} (~{map1_count/total_codes:.1%})")

    map3_count = results_df.filter(
        pl.col("map1_term").is_null() & pl.col("map3_term").is_not_null()
    ).height
    print(f"Mapped by 'ReadcodeList' fallback:    {map3_count:>6} (~{map3_count/total_codes:.1%})")

    map4_count = results_df.filter(
        pl.col("map1_term").is_null() &
        pl.col("map3_term").is_null() &
        pl.col("map4_term").is_not_null()
    ).height
    print(f"Mapped by 'medcodes2' fallback:       {map4_count:>6} (~{map4_count/total_codes:.1%})")
    
    readcode_fallback_count = results_df.filter(
        pl.col("map1_term").is_null() &
        pl.col("map3_term").is_null() &
        pl.col("map4_term").is_null() &
        pl.col("read_code").is_not_null()
    ).height
    print(f"Fell back to just a Read Code:        {readcode_fallback_count:>6} (~{readcode_fallback_count/total_codes:.1%})")
    
    unmapped_count = total_codes - (map1_count + map3_count + map4_count + readcode_fallback_count)
    print(f"Remained completely unmapped:         {unmapped_count:>6} (~{unmapped_count/total_codes:.1%})")
    print("---------------------------------\n")

if __name__ == '__main__':
    analyze_coverage('config.yaml')