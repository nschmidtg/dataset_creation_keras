#!/usr/bin/env python3
import sys
import pandas as pd
sys.path.append('../S3')
from build_model import build
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataframe_dir",
    default="../../results/S4/dataframe.csv",
    help="path to the dataframe generated in S4",
    required=True,
)
parser.add_argument(
    "--save_model_dir",
    default="../../results/S4/model.h5",
    help="path where the model will be saved",
    required=True,
)
parser.add_argument(
    "--epochs",
    default=200,
    help="number of epochs to train the model",
    required=True,
)
parser.add_argument(
    "--batch_size",
    default=50,
    help="number of datapoint used for each epoch",
    required=True,
)
parser.add_argument(
    "--validation_split",
    default=0.2,
    help="proportion of the dataset to be used for validation",
    required=True,
)

def load_dataset(data_file):
    # read dataset
    df = pd.read_csv(data_file)
    # get all columns except target
    X = df.loc[:, df.columns != 'target']
    # from id, filename, start and end
    X = X.drop(X.columns[[0, 1, 2, 3]], axis=1)
    # get target
    Y = df["target"]
    # generate categorical dummies
    Y = pd.get_dummies(Y)

    return (X, Y)

def main(conf):
    epochs = int(conf["epochs"])
    validation_split = float(conf["validation_split"])
    batch_size = int(conf["batch_size"])

    # load dataset
    X, Y = load_dataset(conf["dataframe_dir"])
    # get input shape and pass it to the model
    input_shape = X.shape[1]
    model = build(input_shape)
    # fit model
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    # save model
    model.save(conf["save_model_dir"], save_format='h5')

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    main(arg_dic)
