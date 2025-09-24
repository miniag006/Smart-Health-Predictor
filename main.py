#!/usr/bin/env python3
"""
Debug-friendly main: prints progress and captures exceptions.
"""
import argparse
import traceback
from train import build_and_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SmartHealthPredictor (debug mode)")
    parser.add_argument('--data_dir', type=str, default='/mnt/data', help='Directory containing CSV files')
    parser.add_argument('--symptom_csv', type=str, default='/mnt/data/Symptom-severity.csv', help='Path to symptom severity CSV')
    parser.add_argument('--save', type=str, default='smart_health_pipeline.joblib', help='Path to save trained pipeline')
    args = parser.parse_args()

    print(">>> DEBUG: Starting SmartHealthPredictor training")
    print(">>> DEBUG: data_dir =", args.data_dir)
    print(">>> DEBUG: symptom_csv =", args.symptom_csv)
    print(">>> DEBUG: save path =", args.save)

    try:
        shp = build_and_train(args.data_dir, save_path=args.save, symptom_csv=args.symptom_csv)
        print(">>> DEBUG: build_and_train returned successfully.")
        print(">>> DEBUG: Available selector_info keys:", list(shp.selector_info.keys()))
        print(">>> DEBUG: Saving model to", args.save)
        shp.save(args.save)
        print(">>> DEBUG: Model saved.")
    except Exception as e:
        print(">>> ERROR: Exception occurred during training.")
        traceback.print_exc()
        # Also write the traceback to a file for inspection
        with open("train_error_traceback.txt", "w", encoding="utf-8") as f:
            f.write("Exception during training:\n")
            traceback.print_exc(file=f)
        print(">>> DEBUG: Traceback written to train_error_traceback.txt")
