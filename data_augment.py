import argparse
import logging
import os
import cv2
import shutil
import numpy as np
import imgaug.augmenters as iaa
import pandas as pd


if __name__ == "__main__":
    # define argument here
    args = argparse.ArgumentParser(description="Split dataset as train, valid, and test set, and return csv files")
    args.add_argument("-t", "--train_df_path", default="dataset/train_df.csv", type=str,
                      help="path to the train_df csv file")
    args.add_argument("-s", "--saved_file", default="dataset/train_df_aug.csv", type=str,
                      help="file that store the augment dataframe")
    args.add_argument("-r", "--root_path", default="dataset/train_512", type=str,
                      help="path to the root images directory")
    args.add_argument("-a", "--aug_path", default="dataset/train_aug", type=str,
                      help="path to the augment images directory")
    args = args.parse_args()
    # create augment directory, if it is not exist, and delete old one
    if os.path.exists(args.aug_path):
        shutil.rmtree(args.aug_path)
    shutil.copytree(args.root_path, args.aug_path)
    # define gamma variation range
    aug = iaa.LogContrast((0.9, 1), seed=42)
    # set logging here
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filename="saved/log/data_augment.log")
    logger = logging.getLogger("crop images")
    # load train_df.csv, as I only augment training set, so run dataset_split.py first.
    train_df = pd.read_csv(args.train_df_path, index_col=0)
    # retrieve filenames
    normal_files = train_df[train_df.labels == "N"].filename.values
    disease_files = train_df[train_df.labels != "N"].filename.values
    # set seed
    np.random.seed(42)
    idx_full = np.arange(len(normal_files))
    np.random.shuffle(idx_full)

    # augment number
    aug_num = len(disease_files) - len(normal_files)
    images_ori = [cv2.imread(os.path.join(args.root_path, file), cv2.IMREAD_COLOR)
                  for file in normal_files[idx_full[0:aug_num]]]
    images_aug = aug(images=images_ori)
    # write to augment directory
    for image_aug, index in zip(images_aug, idx_full[0:aug_num]):
        logger.debug(f"Augment file: {normal_files[index]}")
        cv2.imwrite(os.path.join(args.aug_path, f"aug_{normal_files[index]}"), image_aug)
        # add augment images to dataframe
        series = pd.Series({"labels": "N", "filename": f"aug_{normal_files[index]}"})
        train_df = train_df.append(series, ignore_index=True)
    train_df.to_csv(args.saved_file, index=False)
