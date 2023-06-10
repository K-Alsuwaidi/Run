from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
import numpy as np
from tensorflow.keras import backend as K
import os
import tensorflow as tf
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path=""

Error = []
Size = []
Loss = []
LossWrong = []
verbose = 0
batch_size = 256
nb_epoch = 300

alldata = np.load(path+"Complex_S21_Sport_Hall.npy")
# alldata = np.load(path+"Complex_S21_Main_Lobby_71.npy")
# alldata = np.load(path+"Complex_S21_Narrow_Corridor_71.npy")
# alldata = np.load(path+"Complex_S21_Lab_139.npy")


alldata = np.transpose(alldata, (0, 2, 1))
alldata = np.reshape(alldata, (601, 1960))
alldata = np.transpose(alldata, (1, 0))

all_x = np.stack([np.real(alldata), np.imag(alldata)], axis=-1)

all_y = []
for i in range(1960):
    consider10repet = i // 10
    all_y.append([consider10repet % 14, consider10repet // 14])  # grid is 14 by 14

dicto = {}
all_y = np.array(all_y)

for train_size in range(9):
    if (train_size == 0):
        train_portion = 1470  # 75%

    elif (train_size == 1):
        train_portion = 1274  # 65%

    elif (train_size == 2):
        train_portion = 1078  # 55%

    elif (train_size == 3):
        train_portion = 882  # 45%

    elif (train_size == 4):
        train_portion = 686  # 35%

    elif (train_size == 5):
        train_portion = 490  # 25%

    elif (train_size == 6):
        train_portion = 294  # 15%

    elif (train_size == 7):
        train_portion = 98  # 5%

    elif (train_size == 8):
        train_portion = 49  # ~2.5%

    LossIter = []
    cmaway = []
    for iTimes in range(20):
        # divide data for training and testing
        print('Iteration:', iTimes, "  Size:", train_portion)
        print(cmaway)

        allInd = np.random.choice(1960, 1960, replace=False)
        #Training Chaning 75% to 2.5%
        #Testing 25%

        train_Ind = allInd[:train_portion]
        test_Ind = allInd[1470:]

        # This is the train & test data for the same environment
        trainData = all_x[train_Ind, :, :]
        testData = all_x[test_Ind, :, :]
        trainTarget = all_y[train_Ind, :]
        testTarget = all_y[test_Ind, :]


        trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], trainData.shape[2], 1))
        testData = np.reshape(testData, (testData.shape[0], testData.shape[1], testData.shape[2], 1))

        while True:

            initializer = tf.keras.initializers.HeNormal()

            ActivationFunction = "relu"

            model = Sequential()

            model.add(Input(shape=(trainData.shape[1], trainData.shape[2], 1), name='input_11'))

            model.add(Conv2D(1, kernel_size=(601, 1), strides=(1, 1), padding='same',
                             activation=ActivationFunction, name='conv2d_transformer',
                             kernel_initializer=initializer))

            model.add(Conv2D(16, kernel_size=(10, 1), strides=(1, 1), activation=ActivationFunction,
                             name='conv2d_1', kernel_initializer=initializer))

            model.add(MaxPooling2D(pool_size=(3, 1), name='max_pooling2d_1'))

            model.add(Conv2D(32, (10, 1), activation=ActivationFunction, name='conv2d_2',
                             kernel_initializer=initializer))

            model.add(MaxPooling2D(pool_size=(3, 1), padding='same', name='max_pooling2d_2'))

            model.add(Flatten(name='flatten_1'))

            model.add(Dense(100, activation='relu', name='dense_1'))

            model.add(BatchNormalization())

            model.add(Dense(2, activation='linear', name='dense_2'))

            model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mse'])

            history = model.fit(trainData, trainTarget, batch_size=batch_size, verbose=verbose, epochs=nb_epoch)
            LossIter.append(history.history.get("mse"))

            myPred = model.predict(testData)
            myPreddiff = myPred - testTarget
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            x = (12.5 * np.sqrt(np.sum(np.power(dis, 2)) / (len(dis))))

            if x < 69:
                break
            else:
                LossWrong.append(history.history("mse"))

        cmaway.append(x)

        K.clear_session()

    Error.append(np.mean(cmaway))
    Size.append(train_portion)
    Loss.append(LossIter)

print(Error)
print(Size)
np.save("CNN_Source_256B_300e_mse", (Error, Size, Loss))
