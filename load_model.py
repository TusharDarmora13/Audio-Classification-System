from tensorflow.keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from matplotlib import pyplot
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import librosa.display

Genere = ["blues", "classical", "country", "disco", "hiphop", "metal", "pop", "reggae", "rock"]

with open("whole_model_self(specto).json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
k='yes'
while(k!='No' and k!='no'):
    n=input("Enter the [Name] of audio_file:-")
    p=input("Enter the [Path] of audio_file:- ")
    p=p+"/"+n    
    try:
        loaded_model.load_weights("model_weights_self(specto).h5")
        loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        x, sr = librosa.load(p, sr=None) 
        window_size = 1024
        hop_length = 256
        window = np.hanning(window_size) # window size = 1024; hop_length = 256
        stft= librosa.core.spectrum.stft(x, n_fft = window_size, hop_length = hop_length, window = window)
        out = 2 * np.abs(stft) / np.sum(window)
        Xdb = librosa.amplitude_to_db(out, ref = np.max)   
        
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr)
        plt.savefig(r"C:\Users\dogra\Desktop\archive\predict/"+n+'.png')
           
        
        test_image1 = image.load_img(r"C:\Users\dogra\Desktop\archive\predict/"+n+'.png')
        test_image3 = np.array(test_image1)
        img = Image.fromarray(test_image3)
        img.show()


        test_image = image.load_img(r"C:\Users\dogra\Desktop\archive\predict/"+n+'.png', target_size = (288,432))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image/=255.0
        f = loaded_model.predict(test_image)
        print(f)
        f=Genere[np.argmax(f)]
        print("Genre of the audio is = ",f)
    except FileNotFoundError as m:
            print(m)
    
    k=input("Enter No to exit:- ")
    
