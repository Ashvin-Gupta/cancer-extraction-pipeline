# src/utils/mapping_setup.py
import polars as pl
import pandas as pd

def expand_codes(df, code_col, term_col):
    """Helper to expand comma-separated codes into a long format DataFrame."""
    return df.select(
        pl.col(term_col),
        pl.col(code_col).str.split(",").alias("code")
    ).explode("code").with_columns(
        pl.col("code").str.strip_chars(" '\"[]")
    ).filter(pl.col("code").is_not_null() & (pl.col("code") != ""))


def map_all_codes(events_lf: pl.LazyFrame, config: dict) -> pl.LazyFrame:
    """
    Maps raw medcodeids with updated LAB prefix, Read Code filtering,
    and MEASUREMENT prefix logic.
    """
    print("Mapping raw codes with all final cleaning rules...")
    PATHS = config['paths']
    
    # Use Pandas' more lenient CSV reader for the problematic codelist file
    try:
        codelists_pd = pd.read_csv(PATHS['cleaned_codelists'], skip_blank_lines=True)
        codelists_pd.columns = codelists_pd.columns.str.strip()
        codelists_lf = pl.from_pandas(codelists_pd).lazy()
    except Exception as e:
        print(f"FATAL: Could not read the codelist file with Pandas: {e}")
        return events_lf.with_columns(
            code=pl.format("MEDICAL//MAPPING_FAILED//{}", pl.col("raw_code"))
        )
    
    lab_terms = ['MVC','CRP','Hemoglobin','TIBC','HbA1c','plasma_viscosity','ESR','GGT','lymphocyte','platelets','AST','ALP','ferritin','MCH','calcium_serum','neutrophils','h_p_ylori','glucose','cholesterol_triglycerides','bilirubin','anti_ttg','plasma_proteins','BP','amylase','ALT','urea_serum','CA125','creatinine_serum','albumin_serum','WCC','creatinine_urine','iron']

    # Helper function to create a formatted code with the correct LAB/MEDICAL prefix
    def format_code_with_prefix(term_col, code_col):
        return pl.when(pl.col(term_col).is_in(lab_terms)) \
                 .then(pl.format(f"LAB//{{}}//{{}}", pl.col(term_col), pl.col(code_col))) \
                 .otherwise(pl.format(f"MEDICAL//{{}}//{{}}", pl.col(term_col), pl.col(code_col)))

    # --- Create Mapping Tables ---
    map1_lf = expand_codes(codelists_lf, "medcodes", "MedicalTerm").select(
        pl.col("code").alias("raw_code"),
        format_code_with_prefix("MedicalTerm", "code").alias("map1_code")
    ).unique(subset=['raw_code'], keep='first')

    medcode_to_readcode_lf = pl.scan_csv(PATHS['medical_dictionary']).select(
        pl.col("MedCodeId").cast(pl.Utf8).alias("raw_code"),
        pl.col("CleansedReadCode").alias("read_code") 
    ).drop_nulls().unique(subset=['raw_code'], keep='first')

    map3_lf = expand_codes(codelists_lf, "ReadcodeList", "MedicalTerm").select(
        pl.col("code").alias("read_code"),
        format_code_with_prefix("MedicalTerm", "code").alias("map3_code")
    ).unique(subset=['read_code'], keep='first')
        
    map4_lf = expand_codes(codelists_lf, "medcodes2", "MedicalTerm").select(
        pl.col("code").alias("raw_code"),
        format_code_with_prefix("MedicalTerm", "code").alias("map4_code")
    ).unique(subset=['raw_code'], keep='first')

    # --- Perform Sequential Joins ---
    events_lf = events_lf \
        .join(map1_lf, on="raw_code", how="left") \
        .join(medcode_to_readcode_lf, on="raw_code", how="left") \
        .join(map3_lf, on="read_code", how="left") \
        .join(map4_lf, on="raw_code", how="left")
        
    # --- Finalize Mapping with All New Rules ---
    final_lf = events_lf.with_columns(
        code=pl.coalesce(
            pl.col("map1_code"),
            pl.col("map3_code"),
            pl.col("map4_code")
        ).fill_null(
            # This expression is evaluated only for rows where the coalesce result is null
            pl.when(
                pl.col("read_code").is_not_null() &
                ~pl.col("read_code").str.starts_with('0') &
                ~pl.col("read_code").str.starts_with('9') &
                ~pl.col("read_code").str.starts_with("EMI") &
                ~pl.col("read_code").str.starts_with("^ES")
            ).then(
                pl.when(pl.col("numeric_value").is_not_null())
                  .then(pl.format("MEASUREMENT//{}//{}", pl.col("read_code").str.slice(0, 3) + pl.lit("..00"), pl.col("raw_code")))
                  .otherwise(pl.format("MEDICAL//{}//{}", pl.col("read_code").str.slice(0, 3) + pl.lit("..00"), pl.col("raw_code")))
            )
        )
    )

    # --- Drop events that failed all mapping rules (are null) or were filtered out ---
    final_lf = final_lf.filter(pl.col("code").is_not_null())
    
    # Drop all intermediate columns used for mapping
    return final_lf.drop(["map1_code", "read_code", "map3_code", "map4_code"])