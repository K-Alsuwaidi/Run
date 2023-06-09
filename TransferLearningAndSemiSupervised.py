from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import random
from scipy import spatial
import scipy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Error = []
Size = []
Loss = []
alldata = np.load("Complex_S21_Lab_139.npy");
env0 = "Lab"
alldata1 = np.load("Complex_S21_Sport_Hall.npy");
env1 = 'SH'


alldata = np.transpose(alldata, (0, 2, 1))
alldata = np.reshape(alldata, (601, 1960))
alldata = np.transpose(alldata, (1, 0))

alldata1 = np.transpose(alldata1, (0, 2, 1))
alldata1 = np.reshape(alldata1, (601, 1960))
alldata1 = np.transpose(alldata1, (1, 0))

all_x = np.stack([np.real(alldata), np.imag(alldata)], axis=-1)
all_x1 = np.stack([np.real(alldata1), np.imag(alldata1)], axis=-1)


all_y = []
all_y1 = []


for i in range(1960):
    consider10repet = i // 10
    all_y.append([consider10repet % 14, consider10repet // 14])  # grid is 14 by 14

all_y = np.array(all_y)
all_y1 = np.array(all_y)


SourceSize = 1470

for target_size in range(9):
    Error1 = []
    Error2 = []
    Error3 = []
    Error4 = []
    LossSize = []


    if (target_size == 0):
        target_portion = 1470  # 75%

    elif (target_size == 1):
        target_portion = 1274  # 65%

    elif (target_size == 2):
        target_portion = 1078  # 55%

    elif (target_size == 3):
        target_portion = 882  # 45%

    elif (target_size == 4):
        target_portion = 686  # 35%

    elif (target_size == 5):
        target_portion = 490  # 25%

    elif (target_size == 6):
        target_portion = 294  # 15%

    elif (target_size == 7):
        target_portion = 98  # 5%

    elif (target_size == 8):
        target_portion = 49  # ~2.5%

    for Iteration in range(20):
        LossIter = []
        print('Iteration:', Iteration, "  Size:", target_portion)

        allInd = np.random.choice(1960, 1960, replace=False)

        train_Ind = allInd[:1470]
        test_Ind = allInd[1470:]

        train1_Ind = allInd[:target_portion]
        unsup1_Ind = allInd[target_portion:1470]
        test1_Ind = allInd[1470:]

        trainData = all_x[train_Ind, :, :]
        testData = all_x[test_Ind, :, :]
        trainTarget = all_y[train_Ind, :]
        testTarget = all_y[test_Ind, :]

        # This is the train & test data for the different environments
        trainData1_unsupervisied = all_x1[unsup1_Ind, :, :]
        trainData1 = all_x1[train1_Ind, :, :]
        testData1 = all_x1[test1_Ind, :, :]
        trainTarget1 = all_y1[train1_Ind, :]
        testTarget1 = all_y1[test1_Ind, :]

        trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], trainData.shape[2], 1))
        testData = np.reshape(testData, (testData.shape[0], testData.shape[1], testData.shape[2], 1))

        trainData1 = np.reshape(trainData1, (trainData1.shape[0], trainData1.shape[1], trainData1.shape[2], 1))
        trainData1_unsupervisied = np.reshape(trainData1_unsupervisied, (
            trainData1_unsupervisied.shape[0], trainData1_unsupervisied.shape[1],
            trainData1_unsupervisied.shape[2],
            1))
        testData1 = np.reshape(testData1, (testData1.shape[0], testData1.shape[1], testData1.shape[2], 1))


        while True:
            initializer = tf.keras.initializers.HeNormal(seed=None)
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

            batch_size = 32
            nb_epoch = 200  # 200

            history = model.fit(trainData, trainTarget, batch_size=batch_size, verbose=2, epochs=nb_epoch)

            myPred = model.predict(testData)
            myPreddiff = myPred - testTarget
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            x = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / (len(dis)))

            if x < 69:
                break
            else:
                print(x)

        Error1.append(x)
        LossIter.append(history.history.get("mse"))
        model.save("SourceModel")

        print("Source RMSE:", Error1[-1])

        myPred = model.predict(testData1)
        myPreddiff = myPred - testTarget1
        dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
        x = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / (len(dis)))
        Error2.append(x)

        print("Target 1", env1, "no fine tuning:", Error2[-1])

        # Transfer learning target 1

        FineTuneModel1 = tf.keras.models.load_model("SourceModel")

        FineTuneModel1.layers[0].trainable = True
        FineTuneModel1.layers[1].trainable = True
        FineTuneModel1.layers[2].trainable = True
        FineTuneModel1.layers[3].trainable = True
        FineTuneModel1.layers[4].trainable = True
        FineTuneModel1.layers[5].trainable = True
        FineTuneModel1.layers[6].trainable = True
        FineTuneModel1.layers[7].trainable = True
        FineTuneModel1.layers[8].trainable = True



        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mse'])

        history = FineTuneModel1.fit(trainData1, trainTarget1, batch_size=batch_size, verbose=0,
                                     epochs=nb_epoch)

        myPred = FineTuneModel1.predict(testData1)
        myPreddiff = myPred - testTarget1
        dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
        x = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)
        Error3.append(x)
        LossIter.append(history.history.get("mse"))


        print("Target 1", env1, "fine tuning:", Error3[-1])

        FineTuneModel1.save("FineTuneModel")

        if (target_portion < 1470):
            tRMSE4 = []
            kRMSE4 = []
            LossK = []


            trainData1_unsupervisied1 = trainData1_unsupervisied.copy()

            trainData1_unsupervisied1 = np.reshape(trainData1_unsupervisied1,
                                                   [trainData1_unsupervisied1.shape[0],
                                                    trainData1_unsupervisied1.shape[1],
                                                    trainData1_unsupervisied1.shape[2], 1])

            # Proxy label stage
            FineTuneModel2 = tf.keras.models.load_model("FineTuneModel")
            model_output = FineTuneModel2.get_layer("dense_2").output
            m = Model(inputs=FineTuneModel2.input, outputs=model_output)
            proxy_label_data = m.predict(trainData1)
            cur_dim = proxy_label_data.shape[1]

            model_input2 = Input(shape=(cur_dim,), name='input_11')
            Input1 = Dense(40, activation='relu', name='input_layer')(model_input2)
            Intermediate1 = Dense(40, activation='relu', name='intermediate_layer')(Input1)
            Final_output1 = Dense(2, activation='linear', name='output1')(Intermediate1)
            refinment_model = Model(model_input2, Final_output1)
            refinment_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

            refinment_model.fit(proxy_label_data, trainTarget1, batch_size=batch_size, verbose=0,
                                epochs=nb_epoch)
            refinment_model.save('refinemnet')


            for k in range(10, trainData1_unsupervisied1.shape[0], 10):

                Pseudopoints = trainData1_unsupervisied1[:k]

                UnSupModel1 = tf.keras.models.load_model("FineTuneModel")

                UnSupModel1.layers[0].trainable = True
                UnSupModel1.layers[1].trainable = True
                UnSupModel1.layers[2].trainable = True
                UnSupModel1.layers[3].trainable = True
                UnSupModel1.layers[4].trainable = True
                UnSupModel1.layers[5].trainable = True
                UnSupModel1.layers[6].trainable = True
                UnSupModel1.layers[7].trainable = True
                UnSupModel1.layers[8].trainable = True


                UnSupModel1.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse',
                                    metrics=['mse'])

                refinemnet_model1 = tf.keras.models.load_model("refinemnet")
                model_output = FineTuneModel2.get_layer("dense_2").output
                m = Model(inputs=FineTuneModel2.input, outputs=model_output)
                pseudo_label_data = m.predict(Pseudopoints)

                model_output1 = refinemnet_model1.get_layer("output1").output
                m1 = Model(inputs=refinemnet_model1.input, outputs=model_output1)
                pseudo_label_data_refined = m1.predict(pseudo_label_data)

                all_data = np.concatenate((trainData1, Pseudopoints))
                all_data_label = np.concatenate((trainTarget1, pseudo_label_data_refined))

                history = UnSupModel1.fit(all_data, all_data_label, batch_size=batch_size, verbose=0,
                                          epochs=nb_epoch)
                LossIter.append(history.history.get("mse"))

                UnSupModel1.save("UnSupModel")

                myPred = UnSupModel1.predict(testData1)
                myPreddiff = myPred - testTarget1
                dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
                x = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)
                tRMSE4.append(x)
                kRMSE4.append(k)
                LossK.append(history.history)

                print("Target 1", env1, "K:", k, "fine tuning and psudolabel:", tRMSE4[-1])



        Error4.append((tRMSE4, kRMSE4))
        LossIter.append(LossK)
        K.clear_session()

    Error.append((Error1, Error2, Error3, Error4, LossIter))
    print(Error)
    Size.append(target_portion)



Data = (Error, Size, Loss, (env0, env1, 0, 0))
print(Error)
print(Size)
print(Loss)
np.save("SVTL", Data)
