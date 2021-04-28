import essentia
import essentia.standard as estd
import sys
import pandas as pd
from os import listdir, walk
from os.path import join, exists, basename
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_location",
    default="../../results/S1/",
    help="path to the directory where the audio files are",
    required=True,
)
parser.add_argument(
    "--feature_csv_destination",
    default="../../results/S2/input_features.csv",
    help="pathname of the csv what will store the processed input features",
    required=True,
)
parser.add_argument(
    "--frame_size",
    default=200,
    help="number of samples of each frame to analyze",
    required=True,
)

def analyze_sound(audio_path, frame_size=None):
    """Analyze the audio file given in 'sound_path'.
    Use the parameter 'frame_size' to set the size of the chunks in which the audio will
    be split for analysis. If no frame_size is given, the whole audio will be analyzed as
    a single frame.
    [Part of this method was taken from an assignment for the AMPLab course of the Sound and Music
    Computing Master Program. UPF.]
    """
    analysis_output = []  # Here we'll store the analysis results for each chunk (frame) of the audio file

    # Load audio file
    loader = estd.MonoLoader(filename=audio_path)
    audio = loader()

    # Some processing of frame_size parameter to avoid later problems
    if frame_size is None:
        frame_size = len(audio)  # If no frame_size is given use no frames (analyze all audio at once)
    if frame_size % 2 != 0:
        frame_size = frame_size + 1 # Make frame size even

    # Calculate the start and end samples for each equally-spaced audio frame
    frame_start_samples = range(0, len(audio), frame_size)
    frame_start_end_samples = zip(frame_start_samples[:-1], frame_start_samples[1:])

    # Iterate over audio frames and analyze each one
    for count, (fstart, fend) in enumerate(frame_start_end_samples):

        # Get corresponding audio chunk and initialize dictionary to sotre analysis results with some basic metadata
        frame = audio[fstart:fend]
        frame_output = {
            'id': '{0}_f{1}'.format(basename(audio_path), count),
            'path': audio_path,
            'start_sample': fstart,
            'end_sample': fend,
        }

        # Extract loudness
        loudness_algo = estd.Loudness()
        loudness = loudness_algo(frame)
        frame_output['loudness'] = loudness / len(frame)  # Normnalize by length of frame

        # Extract MFCC coefficients
        w_algo = estd.Windowing(type = 'hann')
        spectrum_algo = estd.Spectrum()
        mfcc_algo = estd.MFCC()
        spec = spectrum_algo(w_algo(frame))
        _, mfcc_coeffs = mfcc_algo(spec)
        frame_output.update({'mfcc_{0}'.format(j): mfcc_coeffs[j] for j in range(0, len(mfcc_coeffs))})

        # Extract MalBands coeficients
        melbands_algo = estd.MelBands()
        melbands = melbands_algo(spec)
        frame_output.update({'melbands{0}'.format(j): melbands[j] for j in range(0, len(melbands))})

        # Extract OnsetDetection of the frame
        onset_algo = estd.OnsetDetection()
        onsets = melbands_algo(spec)
        frame_output.update({'onsets{0}'.format(j): onsets[j] for j in range(0, len(onsets))})

        # Extract HPCP
        spectral_peaks_algo = estd.SpectralPeaks()
        HPCP_algo = estd.HPCP()
        HPCP = HPCP_algo(*spectral_peaks_algo(spec))
        frame_output.update({'HPCP{0}'.format(j): HPCP[j] for j in range(0, len(HPCP))})

        # Add frame analysis results to output
        analysis_output.append(frame_output)

    return analysis_output

def main(conf):
    # make sure frame_size is int
    frame_size = int(conf["frame_size"])

    # collect audio dirs
    audio_files = []
    for path, _, files in walk(conf["audio_location"]):
        for name in files:
            audio_files.append(join(path, name))
    print(audio_files)

    # analyze sounds with essentia
    analyses = []
    for audio_file in tqdm(audio_files, desc="processing audio files with essentia"):
        analysis_output = analyze_sound(audio_file, frame_size=frame_size)
        analyses += analysis_output

    # Store analysis results in a new Pandas DataFrame and save it
    df_source = pd.DataFrame(analyses)
    df_source.to_csv(conf["feature_csv_destination"])


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    main(arg_dic)

