# debug_csv.py
import csv
import yaml
import polars as pl

def debug_csv(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    CODELIST_FILE_PATH = PATHS['cleaned_codelists']
    # --- IMPORTANT: Set this to the correct path ---
    # CODELIST_FILE_PATH = '/data/home/qc25022/cancer-extraction-pipeline/src/resources/cleaned_code_lists8.csv'
    
    print(f"--- Inspecting the first 3 lines of: {CODELIST_FILE_PATH} ---\n")
    
    try:
        with open(CODELIST_FILE_PATH, 'r', encoding='utf-8') as f:
            for i in range(3):
                line = f.readline()
                if not line:
                    break
                print(f"Line {i+1}: {line.strip()}")
                # Also try to parse it with Python's built-in CSV reader
                try:
                    parsed = list(csv.reader([line]))
                    print(f"  -> Parsed as {len(parsed[0])} columns: {parsed[0]}\n")
                except Exception as e:
                    print(f"  -> Python's CSV reader failed: {e}\n")
    
    except FileNotFoundError:
        print(f"Error: File not found at '{CODELIST_FILE_PATH}'")
    except Exception as e:
        print(f"An error occurred: {e}")
