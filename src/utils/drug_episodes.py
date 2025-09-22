import glob 
import os
from pathlib import Path
import polars as pl


def create_drug_episodes(prescriptions_df: pl.DataFrame) -> pl.DataFrame:
    """
    Consolidates individual prescription events into START and END events for drug episodes.
   
    """
    if prescriptions_df.is_empty():
        return pl.DataFrame()

    print("  - Consolidating drug episodes...")
    # A drug group is the prescription name, ignoring the specific pack size/formulation
    prescriptions_df = prescriptions_df.with_columns(
        drug_group=pl.col('mapped_code').str.extract(r"^(PRESCRIPTION//[^/]+)")
    ).sort('subject_id', 'drug_group', 'time')

    # Calculate the time difference between consecutive scripts for the same drug
    prescriptions_df = prescriptions_df.with_columns(
        time_diff_days=(pl.col('time').diff().over(['subject_id', 'drug_group'])).dt.days()
    )

    # For each drug, define a cutoff period for a new episode (median + 1 std dev)
    drug_group_stats = prescriptions_df.group_by(['subject_id', 'drug_group']).agg(
        median_diff=pl.col('time_diff_days').median(),
        std_diff=pl.col('time_diff_days').std().fill_null(0) # Fill null for single prescriptions
    )
    
    prescriptions_df = prescriptions_df.join(drug_group_stats, on=['subject_id', 'drug_group'], how='left')
    
    # A new episode starts if the time difference exceeds the cutoff
    prescriptions_df = prescriptions_df.with_columns(
        is_new_episode=pl.when(pl.col('time_diff_days').is_null()) # First script is always a new episode
            .then(True)
            .otherwise(pl.col('time_diff_days') > (pl.col('median_diff') + pl.col('std_diff') + 14)) # Add 14 day buffer
    ).with_columns(
        episode_id=pl.col('is_new_episode').cumsum().over(['subject_id', 'drug_group'])
    )

    # Aggregate by episode to find start and end times
    episodes = prescriptions_df.group_by(['subject_id', 'split', 'drug_group', 'episode_id']).agg(
        start_time=pl.col('time').min(),
        end_time=(pl.col('time').max() + pl.duration(days=pl.col('duration').last().fill_null(60)))
    )

    # Create START events
    start_events = episodes.select(
        pl.col('subject_id'), pl.col('split'),
        pl.col('start_time').alias('time'),
        pl.col('drug_group').str.replace('PRESCRIPTION//', 'START_PRESCRIPTION//').alias('code'),
    )
    # Create END events
    end_events = episodes.select(
        pl.col('subject_id'), pl.col('split'),
        pl.col('end_time').alias('time'),
        pl.col('drug_group').str.replace('PRESCRIPTION//', 'END_PRESCRIPTION//').alias('code'),
    )

    return pl.concat([start_events, end_events])