from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

def data_summary(yes_path, no_path, predSamples):
    positiveSamples = len(listdir(yes_path))
    negativeSamples = len(listdir(no_path))
    predSamples = len(listdir(pred_path))
    samples = positiveSamples + negativeSamples

    positiveSamplesPercentage = (positiveSamples * 100.0) / samples
    negativeSamplesPercentage = (negativeSamples * 100.0) / samples

    print('Data Summary \n')
    print('Number of Images : ', samples)
    print('Number of Images with Label - yes : ', positiveSamples)
    print('Number of Images with Label - no : ', negativeSamples)
    print("Number of unclassified images - ", predSamples)
    print('Percentage of Images with Label - yes : %.2f' % positiveSamplesPercentage, '%')
    print('Percentage of Images with Label - no : %.2f' % negativeSamplesPercentage,'%')

main_path = "./data"
yes_path = main_path + '/yes'
no_path = main_path + '/no'
pred_path = main_path + '/pred'
data_summary(yes_path, no_path, pred_path)

#IMAGE THRESHOLDING
def img_thresholding(image) :
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #thresholding the greyscaled image - cv2.adaptiveThreshold outperforms cv2.THRESH_BINARY
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    #Gives src.type() == CV_8UC1 error
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    #erodes the foregriund boundary
    thresh = cv2.erode(thresh, None, iterations=2)

    #remove small regions of noises
    thresh = cv2.dilate(thresh, None, iterations=2)  

    new_image = thresh

    return new_image

#loading data 
def load_data(dir_list, image_size):

    # load all images in a directory
    images = []
    labels = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in listdir(directory):

            image = cv2.imread(directory + '/' + filename)

            image = img_thresholding(image)

            #cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

            image = image/255
            
            images.append(image)
            if directory[-3:] == 'yes':
                labels.append([1])
            else:
                labels.append([0])
                
    images = np.array(images)
    labels = np.array(labels)
    
    images, labels = shuffle(images, labels)
    
    return images, labels
def pred_load_data(dir_list, image_size):
    # load all images in a directory
    images = []
    image_width, image_height = image_size
    for directory in dir_list:
        for filename in listdir(directory):

            image = cv2.imread(directory + '/' + filename)

            image = img_thresholding(image)

            #cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

            image = image/255            
            images.append(image)            
    predimages = np.array(images)
    return predimages

width, height = (256, 256)

images, labels = load_data([yes_path, no_path], (width, height))
predImage= pred_load_data([pred_path], (width, height))
from tensorflow.keras import backend

if backend.image_data_format() == 'channels_first':
    images = images.reshape(images.shape[0], 1, 256, 256)
    predImage = predImage.reshape(predImage.shape[0], 1, 256, 256)
    input_shape = (1, 256, 256)

else:
    images = images.reshape(images.shape[0], 256, 256, 1)
    predImage = predImage.reshape(predImage.shape[0], 256, 256, 1)   
    input_shape = (256, 256, 1)
    
images = images.astype('float32')
predImage = predImage.astype('float32')
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.2)
print('Number of Training Samples : ', train_images.shape[0])
print('Number of Testing Samples : ', test_images.shape[0])

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=32, epochs=10)

results = model.evaluate(test_images, test_labels, batch_size=1)

testLoss = results[0]
testAccuracy = results[1]*100

print('Test loss : %.2f' % testLoss)
print('Test accuracy : %.2f' % testAccuracy, "%")

predicted_labels = model.predict(test_images)
predicted_labels = (predicted_labels > 0.5)

fmeasureScore = f1_score(test_labels, predicted_labels)
print('F-measure Score : %.2f' % fmeasureScore)

import seaborn as ss
from sklearn.metrics import confusion_matrix

xAxisLabels = ['No Cancer', 'Cancer']
yAxisLabels = ['No Cancer', 'Cancer']

confusionMatrix = confusion_matrix(test_labels, predicted_labels)
ss.heatmap(confusionMatrix, annot = True, xticklabels=xAxisLabels, yticklabels=yAxisLabels)

identificationCorrect = confusionMatrix[0,0] + confusionMatrix[1,1]
identificationIncorrect = confusionMatrix[0,1] + confusionMatrix[1,0]
Identification = identificationCorrect + identificationIncorrect

print('Correct : ', identificationCorrect)
print('Incorrect : ', identificationIncorrect)
print("\n\n\n\n NOW WE SEE PRED_PATH RESULT \n")
newPred = model.predict(predImage)
newPred = (newPred>0.5)
print(newPred)
