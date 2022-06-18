import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from scipy.io import wavfile
import numpy as np

def get_file_paths(dirname):
    file_paths = []  
    for root, directories, files in os.walk(dirname):
        for filename in files:   
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  
    return file_paths    

DIRNAME = r"C:\Users\dogra\Desktop\archive\Data\genres_original\pop"
def main():
    files = get_file_paths(DIRNAME)                 # get all file-paths of all files in dirname and subdirectories
    for file in files:                              # execute for each file
        (filepath, ext) = os.path.splitext(file)    # get the file extension
        file_name = os.path.basename(file)          # get the basename for writing to output file
        if ext == '.wav':                           # only interested if extension is '.wav'
           x, sr = librosa.load(filepath+'.wav', sr=None) 
           window_size = 1024
           hop_length = 256
           window = np.hanning(window_size) # window size = 1024; hop_length = 256
           stft= librosa.core.spectrum.stft(x, n_fft = window_size, hop_length = hop_length, window = window)
           out = 2 * np.abs(stft) / np.sum(window)
           Xdb = librosa.amplitude_to_db(out, ref = np.max)
           plt.figure(figsize=(14, 5))
           librosa.display.specshow(Xdb, sr=sr)
           
           plt.savefig(r'C:\Users\dogra\Desktop\archive\New_Specto\Train\rock/'+file_name+'.png')
           #plt.close()
if __name__ == '__main__':
    main()
    
    
    
'''. A histogram basically depicts an estimate of the probability distribution
 of some variable. To construct a histogram, the range of possible variable values
 gets divided into a series of intervals called bins.     
 
 
spectrogram is a visual representation of the spectrum of frequencies found in
 a signal as they vary with time. Spectrograms of audio frequencies are sometimes
 called voiceprints or voicegrams. When the data is represented in a 3D plot the
 resulting depiction may be called a waterfall. '''