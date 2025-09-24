if __name__ == "__main__":
    import argparse
    from data_utils import list_uploaded_files, read_csv_safe, detect_label_columns
    print("DEBUG: inspect_datasets.py is running")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    print("DEBUG: data_dir =", args.data_dir)
    files = list_uploaded_files(args.data_dir)
    print("DEBUG: Found CSV files:", files)

    if not files:
        print("DEBUG: No CSV files found in dir.")
    
    for p in files:
        print("\n---\nFILE:", p)
        try:
            df = read_csv_safe(p)
            print("shape:", df.shape)
            print("columns:", df.columns.tolist())
            print("first 3 rows:\n", df.head(3).to_string(index=False))
            print("detected label columns:", detect_label_columns(df))
        except Exception as ex:
            print("ERROR reading", p, ":", ex)
