

"""### Building the Model"""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Flatten,Dense,BatchNormalization, Dropout
from keras.optimizers import SGD, Adamax, Adam
import pandas as pd
import csv
from keras.callbacks import Callback


#Conv3D is used for 3D i.e. Videos with time as 3rd dimension
#print(K.image_data_format())
K.set_image_data_format('channels_last')
# Initialize the CNN
classifier = Sequential()

# Adding the layers to CNN

classifier.add(Conv2D(64,(3,3), strides=(1,1),padding='same', input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Conv2D(64,(3,3), padding='same',activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(64,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

# Compiling the CNN
optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
adamax = Adamax(lr=0.001)

classifier.compile(optimizer= 'adam',loss='binary_crossentropy',metrics=['accuracy'])

"""### Training the Model"""

# Fitting the CNN

#Load Existing Model weights and not entire model.
#if we make changes to the model above then load model will overwrite our changes above
#so we only add weights and if the weights correspond to old model then we train from beginning
#we also overwrite the entire saved model when we see improvements in validation loss

import os.path
model_found = False
try :
    if os.path.isfile('CNN_model.h5'):
        classifier.load_weights('CNN_model.h5')
        model_found = True
        print("file found CNN_model.h5. Training will be resumed")
    else:
        print("No Model Weights found. Training from beginning")
except Exception as e:
    print(e)
    print("The model will be trained from beginning")



#defining the callback function
# suppose we had already trained our classifier for 20/25 epochs
# next time if we resume training using saved weights the classifer will not improve much 
# so there is no point in running the classifier for another 25 epochs and hence we
# include an earlystopping callback

#saves the complete model in CNN_model.h5
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
checkpoint = ModelCheckpoint('CNN_model.h5',monitor='val_loss',save_best_only=True,verbose=1, save_weights_only=False)
next_epoch = 0

if model_found:
    try:
        if os.path.isfile('epoch_logs.csv'):
            logfile = pd.read_csv('epoch_logs.csv',header=0)
            next_epoch = int(logfile.iloc[-1,0])+1
            
    except Exception as e:
        print(e)

class CSVLogger2(CSVLogger):
        def __init__(self, filename, separator, append):       
            super(CSVLogger2, self).__init__(filename, separator, append)
           
        def on_epoch_end(self, epoch, logs=None): 
            super(CSVLogger2,self).on_epoch_end(epoch+next_epoch, logs)
            


class EpochHistory(Callback):
    def __init__(self, filename):
        self.filename = filename
        self.fieldnames = ['epoch','start_time','end_time','elapsed_time']
        self.csvfile = None
        if not os.path.isfile(self.filename):  # no existing log file
            self.csvfile = open(self.filename, 'a', newline = '') 
            writer = csv.DictWriter(self.csvfile, fieldnames = self.fieldnames)
            writer.writeheader() 
            self.csvfile.flush()
        else:
            if model_found:
                file_read = open(self.filename, 'r', newline = '') #cannot read/write using same object
                try:
                    if not csv.Sniffer().has_header(file_read.read(1024)): # log file without header
                        self.csvfile = open(self.filename, 'a', newline = '')
                        writer = csv.DictWriter(self.csvfile, fieldnames = self.fieldnames)
                        writer.writeheader()
                        self.csvfile.flush()
                    else:
                        self.csvfile = open(self.filename, 'a', newline = '')  # log file with header
                except Exception as e:
                    if str(e) == 'Could not determine delimiter' : 
                        print("exception: Could not determine delimiter")
                        self.csvfile = open(self.filename, 'a', newline = '')
                        writer = csv.DictWriter(self.csvfile, fieldnames = self.fieldnames)
                        writer.writeheader()
                        self.csvfile.flush()
                    else:
                        raise Exception(str(e))
            else:
                self.csvfile = open(self.filename, 'w', newline = '') 
                writer = csv.DictWriter(self.csvfile, fieldnames = self.fieldnames)
                writer.writeheader() 
                self.csvfile.flush()
                
    def on_train_end(self,logs):
        self.csvfile.close()
            
    def on_epoch_begin(self, epoch, logs):
        self.start_time = time.time()

    def on_epoch_end(self,epoch,logs):
        current_epoch = epoch+ next_epoch
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time        
        writer = csv.DictWriter(self.csvfile, fieldnames = self.fieldnames)
        writer.writerow({'epoch':current_epoch, 'start_time':self.start_time, 
                         'end_time':self.end_time,'elapsed_time':self.elapsed_time})
        self.csvfile.flush()
    
epoch_history = EpochHistory('epoch_history.csv')    

csv_logger = CSVLogger2('epoch_logs.csv', separator=',', append=True) # important for continuous learning        
stopearly = EarlyStopping(monitor='val_loss',min_delta=0, patience=2, verbose=1)
callback_list = [checkpoint, csv_logger, epoch_history]
#callback_list = [checkpoint, stopearly, csv_logger]
#classifier.load_weights('CNN_model.h5')
# Image Augmentation to avoid overfitting
import time
start_time = time.time()
#print(start_time)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),      # same as the expected input_shape
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
# so as to save history
history = classifier.fit_generator(
        training_set,
        steps_per_epoch=(640/32),  # images in training set/batch_size
        epochs=10,
        validation_data=test_set,
        validation_steps=(640/32),
        callbacks = callback_list)  # images in test set/batch_size

# print(history.history)
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

loss = history.history['loss'][-1]
acc = history.history['acc'][-1]
print("Training Accuracy =" +str(acc)) # 0.92525
val_loss = history.history['val_loss'][-1]
val_acc = history.history['val_acc'][-1]
print('Validation/Test accuracy =' + str(val_acc)) #0.7815



"""### Saving the model as json and weights in HDF5 format"""
"""
# serialize model to JSON
model_json = classifier.to_json()
with open("/content/gdrive/My Drive/Colab Notebooks/CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("/content/gdrive/My Drive/Colab Notebooks/CNN_model.h5")
print("Saved model weights and architecture to disk")


"""


"""### Testing on a single image
### Try cat_or_dog_1.jpg / cat_or_dog_2.jpg
"""

# later...
# test on a single test data
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as img
file = 'dataset/single_prediction/cat_or_dog_2.jpg'
Img = img.imread(file)
plt.imshow(Img)
plt.show()
test_image = image.load_img(file,target_size=(64,64))
test_image = image.img_to_array(test_image)

# add a dimension which represents 'm' for number of examples m =1 for single examples
test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)
prediction = classifier.predict_classes(test_image)
#[[1]] for dog,[[0]] for cat 
print(prediction)
classes = training_set.class_indices
#{'cats': 0, 'dogs': 1}
for key, value in classes.items():
    if (value == int(prediction)):
        print("predicted class for the test image is : " + str(key))



"""### Loading the model back from file and verify that prediction is matching"""
"""
# load json and create model
from keras.models import model_from_json
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# returns an uncompiled model instance
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("CNN_model.h5")
print("Loaded model from disk")
"""
from keras.models import load_model
loaded_model = load_model('CNN_model.h5',compile = False)
#find the accuracy of training set & test set which was used earlier
#You must compile a model before training/testing. Use `model.compile(optimizer, loss)`
loaded_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
loaded_model.evaluate(test_image,[[1]])
prediction2 = loaded_model.predict(test_image)
#loaded_model.summary()
for key, value in classes.items():
    if (value == int(prediction2)):
        print("predicted class for the test image is : " + str(key))