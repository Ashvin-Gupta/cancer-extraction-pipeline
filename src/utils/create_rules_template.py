import polars as pl
import pandas as pd
import yaml

def create_rules_template(config_path: str):
    """
    Creates a template CSV file to help build the cleaning_rules.csv.
    It filters the measurement profile for specified lab tests and pre-fills
    known information, leaving the decision columns blank for manual input.
    """
    print("--- Creating Template for cleaning_rules.csv ---")
    
    # --- 1. Load Config and Data ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    OUTPUTS = config['outputs']

    lab_terms = [
        'MVC','CRP','Hemoglobin','TIBC','HbA1c','plasma_viscosity','ESR','GGT',
        'lymphocyte','platelets','AST','ALP','ferritin','MCH','calcium_serum',
        'neutrophils','h_p_ylori','glucose','cholesterol_triglycerides',
        'bilirubin','anti_ttg','plasma_proteins','BP','amylase','ALT','urea_serum',
        'CA125','creatinine_serum','albumin_serum','WCC','creatinine_urine','iron', 'height', 'weight', 'bmi'
    ]
    
    try:
        # profile_df = pl.read_csv("measurement_profile.csv")
        profile_df = pl.read_csv(OUTPUTS['profile_measurement'])
        
        if 'mean' not in profile_df.columns:
            profile_df = profile_df.with_columns(mean=pl.lit(None, dtype=pl.Float64))
        if 'quantile_90' not in profile_df.columns:
            profile_df = profile_df.with_columns(quantile_90=pl.lit(None, dtype=pl.Float64))
            
        numunit_pd = pd.read_csv(
            PATHS['numunit_lookup'],
            sep='\t',
            header=0,
            names=["UnitID", "UnitName"]
        )
        numunit_df = pl.from_pandas(numunit_pd)

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # --- 2. Filter for Lab Tests and Add Unit Names ---
    print("Step 1: Filtering for specified lab tests and joining unit names...")
    
    lab_profile_df = profile_df.filter(pl.col('identifier').is_in(lab_terms))
    
    # --- FIX: Use the correct Polars arguments 'left_on' and 'right_on' ---
    template_df = lab_profile_df.join(numunit_df, left_on="numunitid", right_on="UnitID", how="left")

    # --- 3. Create and Select Columns for the Final Template ---
    print("Step 2: Preparing the template file with blank columns for manual input...")
    
    final_template = template_df.select(
        # Pre-filled columns
        pl.lit("MedicalTerm").alias("IdentifierType"),
        pl.col("identifier").alias("Identifier"),
        pl.col("numunitid").alias("UnitID"),
        pl.col("UnitName"),
        pl.col("mean"),
        
        # Columns for you to fill in
        pl.lit("").alias("TargetUnit"),
        pl.lit("").alias("ConversionFactor"),
        pl.lit("").alias("ValidMin"),
        pl.lit("").alias("ValidMax"),

        # Reference columns
        pl.col("count"),
        pl.col("min"),
        pl.col("max"),
        pl.col("quantile_10"),
        pl.col("quantile_90")
    ).sort("Identifier", "count", descending=[False, True])

    # --- 4. Save the Template File ---
    output_path = OUTPUTS['cleaning_rules_template']
    final_template.write_csv(output_path)

    print("\n--- Template Creation COMPLETE ---")
    print(f"✅ Template file saved to: {output_path}")
    print("You can now open this CSV in a spreadsheet editor to fill in the blank columns.")

if __name__ == '__main__':
    create_rules_template('config.yaml')