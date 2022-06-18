import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, model_from_json 
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, 
        rotation_range=30,  
        zoom_range = 0.20,
        shear_range = 0.25,
   #     brightness_range = (0.2, 0.8),
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( r'C:\Users\dogra\Desktop\archive\1000 data\New_Histogram\Train',
                                                  target_size=(224,224),
                                                  batch_size=128,
                                                  class_mode='categorical',
                                                  color_mode='rgb')

val_set = val_datagen.flow_from_directory(r'C:\Users\dogra\Desktop\archive\1000 data\New_Histogram\Test',
                                            target_size=(224,224),
                                            batch_size=128,
                                            class_mode='categorical',
                                            color_mode='rgb')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding="same", activation="relu", input_shape=[288,432,3]))
cnn.add(tf.keras.layers.BatchNormalization(axis = 3))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'))

cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.BatchNormalization(axis = 3))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.BatchNormalization(axis = 3))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.BatchNormalization(axis = -1))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'))

cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.BatchNormalization(axis = -1))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dropout(0.3))
cnn.add(tf.keras.layers.Dense(units=9, activation='softmax'))

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model1.fit(
          training_set,
          steps_per_epoch=56,
          epochs=70,
          validation_data = val_set,
          validation_steps = 14,
          )
'''
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint])
'''
model_json = model1.to_json()
with open("whole_model_self(Histo MOB).json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model1.save_weights("model_weights_self(Histo MOB).h5")
print("Saved model to disk")


epochs = 70
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()





#PREDECTION PART -----------------------------------------------------------

training_set.class_indices

import numpy as np
from keras.preprocessing import image
import librosa
import librosa.display

p=r"C:/Users/hp/project/Covid/Covid testing/Covids/3cbfec1259635898efae5b57ea3ddea3.jpg"
x, sr = librosa.load(p, sr=None) 
window_size = 1024
hop_length = 256
window = np.hanning(window_size) # window size = 1024; hop_length = 256
stft= librosa.core.spectrum.stft(x, n_fft = window_size, hop_length = hop_length, window = window)
out = 2 * np.abs(stft) / np.sum(window)
Xdb = librosa.amplitude_to_db(out, ref = np.max)   
        
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr)
plt.savefig(r"C:\Users\dogra\Desktop\archive\predict/"+'spe'+'.png')
        
p = r"C:\Users\dogra\Desktop\archive\predict/"+'spe'+'.png'        
test_image = image.load_img(p, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image/=255.0
result = cnn.predict(test_image)
print("---- By CNN ----")
if result == 0:
  print('blues') 
elif result == 1:
  print('classical')
elif result == 2:
  print('country')
elif result == 3:
  print('disco')
elif result == 4:
  print('hiphop')
elif result == 5:
  print('jazz')
elif result == 6:
  print('metal')
elif result == 7:
  print('pop')
elif result == 8:
  print('reggae')
elif result == 9:
  print('rock')


