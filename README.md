# music-genre-classification

We model a classifier to classify songs into different genres. Let us assume a scenario in which, for some reason, we find a bunch of randomly named MP3 files on our hard disk, which are assumed to contain music. Our task is to sort them according to the music genre into different folders such as jazz, classical, country, pop, rock, and metal.

## Dataset

We have used the famous [GITZAN dataset](http://opihi.cs.uvic.ca/sound/genres.tar.gz) for our case study. This dataset was used for the well-known paper in genre classification “ [Musical genre classification of audio signals](https://ieeexplore.ieee.org/document/1021072) “ by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres namely, blues, classical, country, disco, hiphop, jazz, reggae, rock, metal and pop. Each genre consists of 100 sound clips.

## Features

To classify our audio clips, we will choose 5 features, i.e. Mel-Frequency Cepstral Coefficients, Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.

## Tools

We have used Keras API to train a ANN classifier. We have also used [Librosa](https://librosa.github.io/librosa/) module. It is a Python module to analyze audio signals in general but geared more towards music. Other modules are - Pandas, Sci-py, Numpy, PIL, etc.
