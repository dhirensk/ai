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

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,  # images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)  # images in test set




