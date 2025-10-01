import argparse
from src.pipeline.step_01_define_cohort import define_cohort
from src.pipeline.step_02_build_subject_info import build_subject_info
# Import the new Stage 3 scripts
from src.pipeline.step_03a_extract_events import extract_events
from src.pipeline.step_03b_sort_events import sort_events
from src.pipeline.step_03c_process_events import map_and_save_events
from src.pipeline.step_04a_profile_measurements import profile_measurements
from src.pipeline.step_05a_clean_events import clean_events
from src.utils.create_rules_template import create_rules_template
from src.utils.debug_icd10_mapping import debug_mapping
from src.utils.debug_csv import debug_csv
from src.utils.analyse_mappings import analyze_coverage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument(
        "--stage", type=str, required=True, choices=['1', '2', '3', '3a', '3b', '3c', '4', '5', 'debug'],
        help="Which pipeline stage to run."
    )
    args = parser.parse_args()

    if args.stage == '1':
        define_cohort('config.yaml')
    elif args.stage == '2':
        build_subject_info('config.yaml')
    elif args.stage == '3':
        # Run all parts of stage 3 in order
        extract_events('config.yaml')
        sort_events('config.yaml')
        # map_and_save_events('config.yaml')
    elif args.stage == '3a':
        extract_events('config.yaml')
    elif args.stage == '3b':
        sort_events('config.yaml')
    elif args.stage == '3c':
        map_and_save_events('config.yaml')
    elif args.stage == '4':
        # profile_measurements('config.yaml')
        create_rules_template('config.yaml')
    elif args.stage == '5':
        clean_events('config.yaml')
    elif args.stage == 'debug':
        # debug_mapping('config.yaml')
        # debug_csv('config.yaml')
        analyze_coverage('config.yaml')
        