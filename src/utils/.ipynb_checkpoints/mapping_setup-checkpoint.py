# src/utils/mapping_setup.py
import polars as pl

def expand_codes(df: pl.DataFrame, code_col: str, term_col: str) -> pl.DataFrame:
    """
    Expands a DataFrame with comma-separated codes into a long format.
    """
    return df.select(
        pl.col(term_col),
        pl.col(code_col).str.split(",").alias("code")
    ).explode("code").with_columns(
        pl.col("code").str.strip_chars(" '\"[]")
    ).filter(pl.col("code") != "")

# def setup_lookup_tables(config: dict) -> dict:
#     """
#     Loads and pre-processes medical codelists and dictionaries.
#     """
#     print("  - Loading and processing codelists...")
#     PATHS = config['paths']
    
#     codelist = pl.read_csv(PATHS['cleaned_codelists'])

#     medcodes_map = expand_codes(codelist, "medcodes", "MedicalTerm") \
#         .select(
#             pl.col("code").alias("raw_code"),
#             pl.format("MEDICAL//{}//{}", pl.col("MedicalTerm"), pl.col("code")).alias("mapped_code")
#         )

#     snomed_map = pl.read_csv(PATHS['medical_dictionary']) \
#         .select(
#             pl.col("MedCodeId").cast(pl.Utf8).alias("raw_code"),
#             pl.format("MEDICAL//{}//{}", pl.col("CleansedReadCode"), pl.col("MedCodeId")).alias("mapped_code")
#         ).unique(subset=["raw_code"], keep="first")

#     # Return only the relevant mapping tables
#     return {
#         "medcodes_map": medcodes_map,
#         "snomed_map": snomed_map,
#     }

def map_all_codes(events_lf: pl.LazyFrame, config: dict) -> pl.LazyFrame:
    """
    Maps raw medcodeids using a multi-level fallback strategy.
    A flag in the config determines if the fallback stops at SNOMED or continues to ICD-10.
    """
    print("Mapping raw codes using multi-level strategy...")
    PATHS = config['paths']
    
    # --- Level 1: Codelist Mapping (remains the same) ---
    codelist_lf = pl.scan_csv(PATHS['cleaned_codelists'])
    medcodes_long_lf = expand_codes(codelist_lf, "medcodes", "MedicalTerm")

    lab_terms = ['MVC','CRP','Hemoglobin','TIBC','HbA1c','plasma_viscosity','ESR','GGT','lymphocyte','platelets','AST','ALP','ferritin','MCH','calcium_serum','neutrophils','h_p_ylori','glucose','cholesterol_triglycerides','bilirubin','anti_ttg','plasma_proteins','BP','amylase','ALT','urea_serum','CA125','creatinine_serum','albumin_serum','WCC','creatinine_urine','iron']
    
    lab_codes_lf = medcodes_long_lf.filter(pl.col("MedicalTerm").is_in(lab_terms)).select(
        pl.col("code").alias("raw_code"),
        pl.format("LAB//{}//{}", pl.col("MedicalTerm"), pl.col("code")).alias("codelist_mapped")
    )
    other_medical_codes_lf = medcodes_long_lf.filter(~pl.col("MedicalTerm").is_in(lab_terms)).select(
        pl.col("code").alias("raw_code"),
        pl.format("MEDICAL//{}//{}", pl.col("MedicalTerm"), pl.col("code")).alias("codelist_mapped")
    )
    codelist_map_lf = pl.concat([lab_codes_lf, other_medical_codes_lf])
    events_lf = events_lf.join(codelist_map_lf, on="raw_code", how="left")

    # --- Level 2: Fallback Mapping (SNOMED or ICD-10) ---
    
    # This step is always required: map the raw medcode to its SNOMED equivalent.
    medcode_to_snomed_lf = pl.scan_csv(PATHS['medical_dictionary']).select(
        pl.col("MedCodeId").cast(pl.Utf8).alias("raw_code"),
        pl.col("SnomedCTConceptId").cast(pl.Utf8)
    ).drop_nulls().unique(subset=['raw_code'], keep='first')
    
    # Check the flag from the config file to decide the mapping strategy
    if config['study_params']['map_to_icd10']:
        print("Strategy: Mapping medcodes -> SNOMED -> ICD-10")
        
        # Create the second mapping table: SNOMED -> ICD-10
        snomed_to_icd10_lf = pl.scan_csv(PATHS['snomed_icd10_map'], separator='\t').lazy() \
            .filter(pl.col('active') == 1) \
            .sort(['mapGroup', 'mapPriority']) \
            .unique(subset=['referencedComponentId'], keep='first') \
            .select(
                pl.col('referencedComponentId').cast(pl.Utf8).alias('SnomedCTConceptId'),
                pl.col('mapTarget')
            )
        
        # Join the two maps to get a single Medcode -> ICD-10 map
        fallback_map_lf = medcode_to_snomed_lf.join(
            snomed_to_icd10_lf, on="SnomedCTConceptId", how="inner"
        ).select(
            pl.col("raw_code"),
            pl.format("MEDICAL//{}//{}", pl.col("mapTarget"), pl.col("raw_code")).alias("fallback_mapped")
        )
    else:
        print("Strategy: Mapping medcodes -> SNOMED only")
        
        # If not mapping to ICD-10, the fallback map is just the formatted SNOMED code.
        fallback_map_lf = medcode_to_snomed_lf.select(
            pl.col("raw_code"),
            pl.format("MEDICAL//{}//{}", pl.col("SnomedCTConceptId"), pl.col("raw_code")).alias("fallback_mapped")
        )

    # Join the chosen fallback map to our event stream
    events_lf = events_lf.join(fallback_map_lf, on="raw_code", how="left")

    # --- Finalize Mapping ---
    # The coalesce logic works for both strategies
    final_lf = events_lf.with_columns(
        code=pl.coalesce(pl.col("codelist_mapped"), pl.col("fallback_mapped"))
               .fill_null(pl.format("MEDICAL//NULL//{}", pl.col("raw_code")))
    )
    return final_lf.drop(["codelist_mapped", "fallback_mapped"])