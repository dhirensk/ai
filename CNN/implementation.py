#
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Flatten,Dense



#Conv3D is used for 3D i.e. Videos with time as 3rd dimension
#print(K.image_data_format())
K.set_image_data_format('channels_last')
# Initialize the CNN
classifier = Sequential()

# Adding the layers to CNN

classifier.add(Conv2D(32,(3,3),strides=(1,1),padding='valid', input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3), activation = 'relu'))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

# Compiling the CNN

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the CNN
# Image Augmentation to avoid overfitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
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
        steps_per_epoch=(8000/32),  # images in training set/batch_size
        epochs=25,
        validation_data=test_set,
        validation_steps=(2000/32))  # images in test set/batch_size

print(history.history)

loss = history.history['loss'][-1]
acc = history.history['acc'][-1]
print("Training Accuracy =" +str(acc)) # 0.92525
val_loss = history.history['val_loss'][-1]
val_acc = history.history['val_acc'][-1]
print('Validation/Test accuracy =' + str(val_acc)) #0.7815

# serialize model to JSON
model_json = classifier.to_json()
with open("CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("CNN_model.h5")
print("Saved model weights and architecture to disk")
 
# later...
# test on a single test data
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
# add a dimension which represents 'm' for number of examples m =1 for single examples
test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)
prediction = classifier.predict(test_image)
#[[1.0]]

classes = training_set.class_indices
#{'cats': 0, 'dogs': 1}
for key, value in classes.items():
    if (value == int(prediction)):
        print("predicted class for the test image is : " + str(key))

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

#find the accuracy of training set & test set which was used earlier
#You must compile a model before training/testing. Use `model.compile(optimizer, loss)`
loaded_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
loaded_model.evaluate(test_image,[[1]])
loaded_model.summary()


