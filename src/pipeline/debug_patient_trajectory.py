import polars as pl
import yaml
import glob
import os

def debug_patient_trajectory(config_path: str, patient_id: int):
    """
    Isolates a single patient to debug the trajectory window calculation.
    """
    print(f"--- Debugging Trajectory for Patient ID: {patient_id} ---")

    # --- 1. Load Configuration and Subject Info ---
    print("\nStep 1: Loading patient's subject information...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    cancer_type = config['study_params']['cancer_type']
    PATHS = {key: val.format(cancer_type=cancer_type) for key, val in config.get('paths', {}).items()}
    OUTPUTS = {key: val.format(cancer_type=cancer_type) for key, val in config.get('outputs', {}).items()}

    try:
        subject_info = pl.read_csv(OUTPUTS['subject_information_file']) \
            .filter(pl.col('subject_id') == patient_id)
        
        if subject_info.is_empty():
            print(f"Error: Patient ID {patient_id} not found in subject_information.csv")
            return
            
        is_case = subject_info.get_column('is_case')[0]
        cancer_date_str = subject_info.get_column('cancerdate')[0]
        
        print(f"  - Patient Status: {'Case' if is_case == 1 else 'Control'}")
        print(f"  - Cancer Date from file: {cancer_date_str}")
        
    except Exception as e:
        print(f"Error loading subject information: {e}")
        return

    # --- 2. Load ALL Events for This Patient ---
    print("\nStep 2: Loading all raw events for this patient...")
    DATE_FORMAT = "%d/%m/%Y"
    observation_files = glob.glob(os.path.join(PATHS['observation_data_dir'], "*.txt"))
    
    observation_dtypes = {
        "e_patid": pl.Int64,
        "obsdate": pl.String,
        "medcodeid": pl.String,
    }

    patient_events_lf = pl.concat(
        [
            pl.scan_csv(
                f, 
                separator="\t", 
                has_header=True,
                schema_overrides=observation_dtypes # Enforce types for the columns we read
            )
            .filter(pl.col('e_patid') == patient_id)
            .select(observation_dtypes.keys()) # Only select the columns we need
            for f in observation_files
        ],
        how="vertical"
    ).with_columns(
        time=pl.col("obsdate").str.to_date(DATE_FORMAT, strict=False)
    )
    
    patient_events_df = patient_events_lf.collect().drop_nulls('time')
    
    if patient_events_df.is_empty():
        print("No observation events found for this patient.")
        return
        
    min_event_date = patient_events_df.get_column('time').min()
    max_event_date = patient_events_df.get_column('time').max()
    print(f"  - Patient's first event date: {min_event_date}")
    print(f"  - Patient's last event date:  {max_event_date}")

    # --- 3. Apply the EXACT Same Trajectory Logic ---
    print("\nStep 3: Applying trajectory window logic...")
    
    # Re-create the small subjects_lf for the join
    subjects_lf = subject_info.lazy().rename({"subject_id": "e_patid"}) \
        .with_columns(cancerdate=pl.col("cancerdate").str.to_datetime().cast(pl.Date))

    # Join context and apply window functions
    calculated_df = patient_events_df.lazy().join(subjects_lf, on="e_patid", how="inner") \
        .with_columns(last_event_date=pl.max("time").over("e_patid")) \
        .with_columns(
            end_date=pl.when(pl.col("is_case") == 1)
                       .then(pl.col("cancerdate"))
                       .otherwise(pl.col("last_event_date").dt.offset_by("-1y")),
            start_date=pl.when(pl.col("is_case") == 1)
                        .then(pl.col("cancerdate").dt.offset_by("-5y"))
                        .otherwise(pl.col("last_event_date").dt.offset_by("-6y"))
        ).collect()

    if calculated_df.is_empty():
        print("Could not calculate trajectory window. The join might have failed.")
        return

    # --- 4. Report the Calculated Dates ---
    final_start_date = calculated_df.get_column('start_date')[0]
    final_end_date = calculated_df.get_column('end_date')[0]
    
    print("\n--- DEBUG RESULTS ---")
    print(f"Calculated Start Date for filtering: {final_start_date}")
    print(f"Calculated End Date for filtering:   {final_end_date}")

    # --- 5. Show Final Filtered Trajectory ---
    final_trajectory = calculated_df.filter(
        pl.col("time").is_between(pl.col("start_date"), pl.col("end_date"))
    )
    
    if not final_trajectory.is_empty():
        final_min_date = final_trajectory.get_column('time').min()
        final_max_date = final_trajectory.get_column('time').max()
        duration_days = (final_max_date - final_min_date).days
        
        print(f"\nFinal trajectory starts on: {final_min_date}")
        print(f"Final trajectory ends on:   {final_max_date}")
        print(f"Total duration of kept events: {duration_days} days (~{duration_days / 365.25:.1f} years)")
    else:
        print("\nNo events were kept after filtering.")
    print("--------------------------")


if __name__ == '__main__':
    # --- EDIT THIS LINE ---
    PATIENT_TO_DEBUG = 362864450976 
    
    debug_patient_trajectory(
        config_path='config.yaml',
        patient_id=PATIENT_TO_DEBUG
    )