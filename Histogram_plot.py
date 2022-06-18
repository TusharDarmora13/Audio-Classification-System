import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from scipy.io import wavfile

def get_file_paths(dirname):
    file_paths = []  
    for root, directories, files in os.walk(dirname):
        for filename in files:   
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  
    return file_paths    
#C:\Users\dogra\Desktop\archive\Data\genres_original\blues
DIRNAME = r"C:\Users\dogra\Desktop\archive\NEW_DATA\rock"
def main():
    files = get_file_paths(DIRNAME)                 # get all file-paths of all files in dirname and subdirectories
    for file in files:                              # execute for each file
        (filepath, ext) = os.path.splitext(file)    # get the file extension
        file_name = os.path.basename(file)          # get the basename for writing to output file
        if ext == '.wav':                           # only interested if extension is '.wav'
           rate, data = wavfile.read(filepath+'.wav') # reading wave file.
           #print ('All_data =',data)
           #print('Number of sample in DATA =',len(data))
           #c=data[0:499]            # reading first 500 samples from data variable with contain 200965 samples.
           plt.figure(figsize=(14, 5))
           fig = plt.hist(data, bins='auto')  # arguments are passed to np.histogram.
           plt.savefig(r'C:\Users\dogra\Desktop\archive\1000 data\New_Histogram\Train\rock/'+file_name+'.png')
           #plt.close()
if __name__ == '__main__':
    main()
    
    
#face color and edge color taking white by default.

rate, data = wavfile.read(r"C:\Users\dogra\Desktop\archive\Data\genres_original\blues\blues.00002.wav") # reading wave file.
plt.figure(figsize=(14, 5))
plt.hist(data, bins = "auto")
#plt.show()
plt.savefig(r'C:\Users\dogra\Desktop\archive\1000 data\New_Histogram\Train\blues/hj1'+'.png', facecolor='auto', edgecolor = 'auto')

'''
x, sr = librosa.load(r"C:\Users\dogra\Desktop\archive\Data\genres_original\blues\blues.00002.wav", sr=None)
plt.hist(x, bins = "auto")
plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr)
plt.savefig(r'C:\Users\dogra\Desktop\archive\1000 data\New_Histogram\Train\blues/hj5'+'.png')
'''


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 100, 5)
y = np.sin(x)

plt.hist(data)
plt.savefig(r'C:\Users\dogra\Desktop\archive\1000 data\New_Histogram\Train\blues/hj5'+'.png')
