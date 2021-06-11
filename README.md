# Loudness Classificator
Proyect that creates a dataset based on metadata written in a SQLite database, extract audio features using Essentia, and use them to train a Keras Fully Connected Neural Network for classification in terms of Loudness
## instalation

```source venv/bin/activate```

```pip3 install -r requirements.txt```


## S1
transcode_audio_files.py takes the information stored in the sqlite database and and a query text to get the filtered audio files paths. It also receives sample_rate to resample the audio and an output format to write the files in mono, 16 bits, with the desired sample_rate.
I used torchaudio since is more flexible than librosa when working with .mp3 files, in case that the audio files include some.

for using it:

```cd code/S1```

```python transcode_audio_files.py --database_location=/mnt/disks/data/loudsenseDB.db --data_destination=../../results/S1 --database_query='SELECT path FROM recording WHERE reviewed = 1;' --resample=8000 --output_format=wav```

## S2
Here I used essentia to analyze the features of frames of customizable length. The analyzed features were:

- Loudness
- MFCC
- MelBands
- OnsetDetection
- HPCP

```cd code/S2```

```python generate_input_features.py --audio_location=../../results/S1 --feature_csv_destination=../../results/S2/input_features.csv --frame_size=200```

## S3
The model is a basic fully connected neural network. I think this is my weak point: since we are working with numerical features for each frame, it did not make sense for me to make 2D convolutions. The main reason for this is that each datapoint is an independent audio frame with the respective features (1D). Maybe I missed something, but I would love to have some feedback about how 2D convolutions + maxpooling could be used for this task.

## S4
For this part I created an empty dataframe with the respective columns and since I was iterating over the annotations and comparing them with my frame-feature dataframe, I was adding a new block. This procedure ensures that no datapoints without annotations are added to the dataset.
For this scipt is imporant to pass the sample sample_rate used in transcode_audio_files.py in order to comparte the start and end frames correcly.

```cd code/S4```

```python create_dataset.py --input_features=../../results/S2/input_features.csv --annotations=/mnt/disks/data/loudsense-annotations.csv --out_csv=../../results/S4/dataframe.csv --sample_rate=8000```

## S5

I decided to encapsulate the code in a main method in order to pass the paths of the dataset and the path to save the model through the terminal

```cd code/S5```

```python train.py --save_model_dir=../../results/S4/model.h5 --dataframe_dir=../../results/S4/dataframe.csv --batch_size=50 --epochs=200 --validation_split=0.2```
