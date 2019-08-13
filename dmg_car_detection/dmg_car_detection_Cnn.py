# Convolutional Neural Network

# Installing Keras
# pip install --upgrade keras

#Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

#we can change the input shape and max pooling shape for better accuracy but the training time will increase.

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images

#The below code is from keras documentation

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 3033,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 768)

classifier.summary()

# Saving weights
fname = (r"Filepath_where_you_want_to_save_the_model\dmg_car-weights-CNN.h5")
classifier.save(fname)

from keras.models import load_model
# Loading weights
fname = (r"Filepath_where_you_want_to_save_the_model\dmg_car-weights-CNN.h5")
new1 = load_model(fname)
classifier.summary()

# Predicting a new image using our CNN model
import numpy as np
from keras.preprocessing import image

#Target size should be same as the input shape of the cnn
test_image = image.load_img(r"file_path_for_the_image\d_img1.jpg", target_size=(64,64,3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = new1.predict(test_image)

if result[0][0] >= 0.7:
    prediction = 'good'
else:
    prediction = 'damaged'
print (prediction)




