import os
import matplotlib.pyplot as plt
#for loading and visualizing audio files
import librosa
import librosa.display
#to play audio
import IPython.display as ipd
from scipy import signal
from scipy.io import wavfile


audio_fpath = r"C:/Users/dogra/Desktop/archive/Data/genres_original/blues/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))


#Step-2: Load audio file and visualize its waveform (using librosa)

x, sr = librosa.load(audio_fpath+audio_clips[2], sr=44100)
print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


#Step-3: Convert the audio waveform to spectrogram

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


#Applying log transformation on the loaded audio signals

plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


