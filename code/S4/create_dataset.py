import sys
from os import listdir, makedirs, walk
from os.path import join, exists, basename
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_features",
    default="../../results/S2/input_features.csv",
    help="path to the directory where the csv with the essentia features are",
    required=True,
)
parser.add_argument(
    "--annotations",
    default="/mnt/disk/data/loudsense-annotations.csv",
    help="path to the directory where the csv with the annotations are",
    required=True,
)
parser.add_argument(
    "--out_csv",
    default="../../results/S4/dataframe.csv",
    help="filename of the generated csv dataframe",
    required=True,
)
parser.add_argument(
    "--sample_rate",
    default=8000,
    help="sample_rate of the audios processed by S2",
    required=True,
)

class_dict = {
    "audible": 2,
    "barely": 1,
    "inaudible": 0,
}

def main(conf):
    # make sure sample_rate is int
    sample_rate = int(conf["sample_rate"])
    # read dataframes
    df_features = pd.read_csv(open(conf["input_features"]))
    df_target = pd.read_csv(open(conf["annotations"]))
    # create empty target column in df_features
    columns = list(df_target.columns)
    columns.append("target")

    # empty dataset
    df = pd.DataFrame(columns=columns)
    # iterate over the target dataframes inserting the target label
    for index, row in tqdm(df_target.iterrows(), desc="processing files"):
        # convert seconds in annotations to frame using the sample_rate
        start_frame = int(row["start"] * sample_rate)
        end_frame = int(row["end"] * sample_rate)
        regex_query = '^' + row['file_name']
        # filter by filename
        file_filter = df_features[df_features.id.str.contains(regex_query)]

        # file by frame start and end
        sample_filter = file_filter[file_filter["start_sample"] >= start_frame]
        sample_filter = sample_filter[sample_filter["end_sample"] <= end_frame]

        # add column
        sample_filter["target"] = class_dict[row["class"]]
        df = pd.concat([df, sample_filter], ignore_index=True)

    # clean up columns of annotation dataset
    df = df.drop(df.columns[[0, 1, 2, 3, 5]], axis=1)
    # write csv
    df.to_csv(conf["out_csv"])

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    main(arg_dic)
