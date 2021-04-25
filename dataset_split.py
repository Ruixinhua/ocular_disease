import argparse
import os

import pandas as pd
import numpy as np


def statistic(df, kind):
    print(f"{kind} info:\n Count: {len(df)}, positive label(N): {len(df[df.labels == 'N'])}, " +
          f"negative label(not N): {len(df[df.labels != 'N'])}")


if __name__ == "__main__":
    # define argument here
    args = argparse.ArgumentParser(description="Split dataset as train, valid, and test set, and return csv files")
    args.add_argument("-f", "--full_df_path", default="dataset/full_df.csv", type=str,
                      help="path to the full_df csv file")
    args.add_argument("-v", "--valid_ratio", default=0.15, type=float,
                      help="the ratio of validation set")
    args.add_argument("-t", "--test_ratio", default=0.15, type=float,
                      help="the ratio of test set")
    args.add_argument("-s", "--saved_dir", default="dataset/", type=str,
                      help="the directory that save csv files of train, valid, and test set")
    args = args.parse_args()
    # initial dataframe of train, valid ,and test set.
    train_df = pd.DataFrame(columns=["ID", "labels", "filename"])
    valid_df, test_df = train_df.copy(), train_df.copy()
    full_df = pd.read_csv(args.full_df_path)
    # the labels in full_df is a string list like "['N']", take it off
    full_df.labels = full_df.labels.apply(lambda i: i[2])
    # set seed to ensure it can be reproduced
    np.random.seed(42)
    # split dataset according to sub-class, as the images number of sub-class is various, see dataset_analysis notebook
    for label, group in full_df.groupby(["labels"]):
        # reset index of group
        group = group.reset_index()
        # get the full index and shuffle it
        idx_full = np.arange(len(group))
        np.random.shuffle(idx_full)
        # calculate the length of valid and test set
        test_len = round(len(idx_full) * args.test_ratio)
        valid_len = round(len(idx_full) * args.valid_ratio)
        # put the values into train, valid, and test dataframe
        test_df = test_df.append(group.loc[pd.Index(idx_full[0:test_len]), test_df.columns.values], ignore_index=True)
        valid_df = valid_df.append(
            group.loc[pd.Index(idx_full[test_len:valid_len+test_len]), valid_df.columns.values], ignore_index=True
        )
        train_df = train_df.append(
            group.loc[pd.Index(idx_full[valid_len+test_len:]), train_df.columns.values], ignore_index=True
        )
    # print some statistic information
    statistic(train_df, "Training set")
    statistic(valid_df, "Validation set")
    statistic(test_df, "Test set")
    # save dataframe to the path
    train_df.to_csv(os.path.join(args.saved_dir, "train_df.csv"))
    valid_df.to_csv(os.path.join(args.saved_dir, "valid_df.csv"))
    test_df.to_csv(os.path.join(args.saved_dir, "test_df.csv"))
