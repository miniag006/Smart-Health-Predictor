import argparse, os, pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if 'prognosis' not in df.columns and 'prognosis ' in df.columns:
        df.rename(columns={'prognosis ': 'prognosis'}, inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def main(train_path, test_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_df = load_and_clean(train_path)
    test_df = load_and_clean(test_path)


    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]


    
    feat_cols = [c for c in train_df.columns if c!='prognosis']
    for c in feat_cols:
        if c not in test_df.columns:
            test_df[c] = 0
    
    train_df = train_df[feat_cols + ['prognosis']]
    test_df = test_df[feat_cols + ['prognosis']]

    
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    
    train_df.to_csv(os.path.join(out_dir, "prepared_train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "prepared_test.csv"), index=False)
    print("Saved prepared_train.csv and prepared_test.csv in", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="Training.csv")
    parser.add_argument("--test", default="Testing.csv")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    main(args.train, args.test, args.out)
