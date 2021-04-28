import sys
from os import listdir, makedirs, walk
from os.path import join, exists, basename
import math
from shutil import copyfile
import torch
import torchaudio
import sqlite3
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database_location",
    default="/mnt/disk/data/loudsenseDB.db",
    help="path to the sqlite database where the audio paths and status is detailed",
    required=True,
)
parser.add_argument(
    "--data_destination",
    default="results/S1",
    help="path where the transcoded files will be stored",
    required=True,
)
parser.add_argument(
    "--database_query",
    default="SELECT path FROM recording WHERE reviewed = 1;",
    help="query text to query the database",
    required=True,
)
# optional parameters
parser.add_argument(
    "--output_format",
    default="wav",
    help="format of the desired files. If not provided original format "
    "is preserved",
)
parser.add_argument(
    "--resample",
    default=8000,
    help="specify with int value to resample the audio files. If not "
    "provided, original sample_rate is preserved",
)

def add_audio_to_dataset(audio_dir, destination_dir, destination_sr):
    """Resamples and saves or copies the audio_dir to the destination_dir
        only if it is not yet in the directory

    Parameters
    ----------
    audio_dir : str
        path to the audio file
    destination_dir: str
        path os the destination file with extension
    destination_sr: None or int
        None if resample is not required or (int) the destination
        sample rate.
    """
    if not exists(destination_dir):
        if not destination_sr == None:
            # resample and save

            # make sure destination sample_rate is int
            destination_sr = int(destination_sr)
            # load audio
            audio, original_sr = torchaudio.load(audio_dir)
            # make mono
            if not audio.shape[0] == 1:
                audio = torch.mean(audio, dim=0).unsqueeze(0)
            # resample audio
            if not original_sr == destination_sr:
                audio = torchaudio.transforms.Resample(
                    original_sr,
                    destination_sr
                )(audio)
            # save audio
            torchaudio.save(
                filepath=destination_dir,
                src=audio,
                sample_rate=destination_sr,
                bits_per_sample=16
            )
        else:
            # only copy file
            copyfile(audio_dir, destination_dir)

def create_connection(db_file):
    """Create a database connection to the SQLite database
        specified by the db_file

    -----------
    db_file : database file
    :return : Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def main(conf):
    # first of all query the database to obtain the paths
    # of the reviewed files
    conn = create_connection(conf["database_location"])
    cur = conn.cursor()
    cur.execute(conf["database_query"])
    rows = cur.fetchall()

    # add the results of the query to an array
    audio_files = []
    for row in rows:
        audio_files.append(row[0])

    # iterate over the audios resampling or copying the audio.
    # The new versions are then written in the respective directory
    # inside the data_destination directory.
    for audio_dir in tqdm(audio_files, desc="processed files"):
        # change the output format
        if not conf["output_format"] == None:
            file_name = basename(audio_dir).split('.')[0] + '.' + conf["output_format"]
        else:
            file_name = basename(audio_dir)

        # copy or resample and save the file in the respective set
        destination = join(conf["data_destination"], file_name)
        add_audio_to_dataset(audio_dir, destination, conf["resample"])

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    main(arg_dic)
