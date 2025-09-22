# debug_icd10_mapping.py

import polars as pl
import yaml

def debug_mapping(config_path):
    """
    Performs a step-by-step check of the two-stage mapping process:
    1. Finds which raw medcodeids map to a SNOMED code.
    2. Finds which of those SNOMED codes map to an active ICD-10 code.
    """
    print("--- Debugging Two-Step (Medcode -> SNOMED -> ICD-10) Mapping ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    OUTPUTS = config['outputs']

    # --- 1. Get a sample of raw codes from your data that need mapping ---
    print("\nStep 1: Getting a sample of raw codes from your sorted events...")
    raw_codes_to_check = pl.scan_parquet(f"{OUTPUTS['intermediate_sorted_dir']}/*.parquet") \
        .filter(pl.col('code') != "MEDS_BIRTH") \
        .select(
            raw_code=pl.col("code").str.extract(r"//(.*)$", 1).cast(pl.Utf8)
        ) \
        .drop_nulls() \
        .unique() \
        .head(10000) # Check the first 10,000 unique codes
    
    df_codes_to_check = raw_codes_to_check.collect()
    print(f"Found {len(df_codes_to_check)} unique raw codes to check.")

    # --- 2. Load the two mapping tables ---
    print("\nStep 2: Loading the Medcode->SNOMED and SNOMED->ICD10 maps...")
    
    # Map 1: Medcode -> SNOMED
    medcode_to_snomed = pl.read_csv(PATHS['medical_dictionary']).select(
        pl.col("MedCodeId").cast(pl.Utf8).alias("raw_code"),
        pl.col("SnomedCTConceptId").cast(pl.Utf8)
    ).drop_nulls().unique(subset=['raw_code'], keep='first')
    print(f"Loaded {len(medcode_to_snomed)} Medcode-to-SNOMED rules.")

    # Map 2: SNOMED -> ICD-10
    snomed_to_icd10 = pl.read_csv(PATHS['snomed_icd10_map'], separator='\t') \
        .filter(pl.col('active') == 1) \
        .sort(['mapGroup', 'mapPriority']) \
        .unique(subset=['referencedComponentId'], keep='first') \
        .select(
            pl.col('referencedComponentId').cast(pl.Utf8).alias('SnomedCTConceptId'),
            pl.col('mapTarget')
        )
    print(f"Loaded {len(snomed_to_icd10)} active SNOMED-to-ICD10 rules.")

    # --- 3. Perform a two-step INNER JOIN to find successful matches ---
    print("\nStep 3: Performing two-step INNER join to find end-to-end matches...")
    
    # First, find which of our codes have a SNOMED equivalent
    snomed_matches = df_codes_to_check.join(medcode_to_snomed, on="raw_code", how="inner")
    
    # Second, find which of those SNOMED codes have an active ICD-10 map
    final_matches = snomed_matches.join(snomed_to_icd10, on="SnomedCTConceptId", how="inner")

    # --- 4. Report the results ---
    print("\n--- DEBUG RESULTS ---")
    print(f"Initial unique codes to check: {len(df_codes_to_check)}")
    print(f"Codes with a SNOMED mapping: {len(snomed_matches)}")
    print(f"Codes that completed the full mapping to ICD-10: {len(final_matches)}")

    if final_matches.is_empty():
        print("\n❌ Found 0 end-to-end matches.")
        if not snomed_matches.is_empty():
            print("This means that while some of your medcodes map to SNOMED codes, those SNOMED codes do not have an active ICD-10 mapping in your file.")
        else:
            print("This means none of your medcodes could be mapped to a SNOMED code in the first step.")
    else:
        print(f"\n✅ Success! Found {len(final_matches)} complete matches. The mapping logic is working.")
        meaningful_matches = final_matches.filter(
            ~pl.col('mapTarget').is_in(["#NIS", "#NC"])
        )
        print(f"  -> Of these, {len(meaningful_matches)} map to a meaningful ICD-10 code (not '#NIS' or '#NC').")
        print(meaningful_matches.head(30))
        print("Here is a sample of the successful matches:")
        print(final_matches.head(30))

if __name__ == '__main__':
    debug_mapping('config.yaml')