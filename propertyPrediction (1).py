# Standard library imports
import os
import sys

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from scipy.stats import boxcox
# Keras imports
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization, GlobalAveragePooling2D)
from keras.metrics import (
    MeanSquaredError, RootMeanSquaredError)
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2, ResNet50V2
from keras.optimizers import Adam
# FROM FILE
from tensorflow.python.client import device_lib
from keras import backend as K

if __name__ == "__main__":
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    
    #df = pd.read_csv("/home/tusharsoodd/alexanderaOnlYStableWithCorrectFileNames.csv")
    df = pd.read_csv("/home/tusharsoodd/MPTRY3CSV.csv")
    column_names = df.columns
        
    columnName = sys.argv[1]
    if str(sys.argv[2]) == "none" and str(sys.argv[3]) == "none":
        minCutoff = df[columnName].min()
        maxCutoff = df[columnName].max()
    elif str(sys.argv[2]) == "none" and not str(sys.argv[3]) == "none":
        minCutoff = float(sys.argv[2])
        maxCutoff = df[columnName].max()
    elif not str(sys.argv[2]) == "none" and str(sys.argv[3]) == "none":
        minCutoff = df[columnName].min()
        maxCutoff = float(sys.argv[3])
    else:
        minCutoff = float(sys.argv[2])
        maxCutoff = float(sys.argv[3])
    modelSaveName = sys.argv[4]
    minimum_value = df[columnName].min()
    
    #df[columnName] = df[columnName].apply(np.sqrt)

   

    if sys.argv[6] == 'sqrt':
        print("sq rting")
        df[columnName] = np.sqrt(df[columnName])
    elif sys.argv[6] == 'log':
        print("LOGGING")
        df[columnName] = np.log(df[columnName])
    else:
        pass
   

    if sys.argv[5] == 'true':
        print("STANDARDIZING")
        mean = df[columnName].mean()
        std_dev = df[columnName].std()
        df[columnName] = (df[columnName] - mean) / std_dev

    


    df_filtered = df[(df[columnName] > minCutoff) & (df[columnName] < maxCutoff) & (~df[columnName].isna()) ]

    df_filtered['filename'] = df_filtered['filename'].apply(lambda x: "/home/tusharsoodd/TRY3DUMP/" + x)
    print(f"ST DEV OF COLUMN: {df_filtered[columnName].std()}")
    
    train_df, test_df = train_test_split(df_filtered, train_size=0.8, shuffle=True, random_state=1)
    print("defining generators")
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        dtype='float64'
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        dtype='float64'
    )
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    print("provisioning images")
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col=columnName,
        target_size=(64, 64),
        color_mode='rgb',
        class_mode='raw',
        batch_size=8,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col=columnName,
        target_size=(64, 64),
        color_mode='rgb',
        class_mode='raw',
        batch_size=8,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col=columnName,
        target_size=(64, 64),
        color_mode='rgb',
        class_mode='raw',
        batch_size=8,
        shuffle=True
    )

    true_labels = test_images.labels

    inputs = tf.keras.Input(shape=(64, 64, 1))
    tf.keras.backend.set_floatx('float64')

    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64,64,3))
    
    for layer in base.layers:
        layer.trainable = False
    

    mmodel = Sequential([
    base,
    #Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(64, 64, 1)),
    #BatchNormalization(),
    #MaxPooling2D((2, 2)),
    #Dropout(.25),
    #Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    #BatchNormalization(),
    #MaxPooling2D((2, 2)),
    #Dropout(.25),
    #Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    #BatchNormalization(),
    #MaxPooling2D((2, 2)),
    #Dropout(.25),
    GlobalAveragePooling2D(),
    Dense(1280, activation='leaky_relu'),
    Dropout(.2),
    Dense(1280, activation='leaky_relu'),
    Dropout(.2),
    Dense(1280, activation='leaky_relu'),
    Dropout(.2),
    Dense(1)  # Output layer for regression
])
    del mmodel 
    checkpoint = ModelCheckpoint(filepath=f"{modelSaveName}.keras", 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)   # Output layer for regression
    ])
    #del model
    #model=mmodel

    print("compiling model")
    print(model.summary())
    def cyclical_lr_np(epoch, lr):
        cycle = np.floor(1 + (iter / (2*step_size)))
        x = np.abs((iter/step_size) - 2*cycle + 1)
        lr = base_lr + (max_lr-base_lr) * (np.max((0, (1-x))))

        return lr
    def cyclical_learning_rate(epoch, lr):
            base_lr = 0.0005
            max_lr = 0.01
            step_size = 11514
            gamma = 0.90
            cycle = np.floor(1 + epoch / (2 * step_size))
            x = np.abs(epoch / step_size - 2 * cycle + 1)
            new_lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma**epoch
            return new_lr
    lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
                .0001, 10000)
    optimizer =Adam(.0001) #lr_decayed_fn)
    # Step 4: Compile your model
    model.compile(optimizer,
                loss='mape',  # Mean Absolute Error for regression
                metrics=[keras.metrics.MeanSquaredError(), keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    print("training model")
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=1)  # Step 5: Train your model
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=20,
        verbose=1,
        callbacks=[callback, checkpoint]
    )


    model.save(f"{modelSaveName}.keras")
    testtest=model.evaluate(test_images)

    #y_pred = model.predict(test_images)
    #error = np.abs(true_labels - y_pred)

    #plt.figure(figsize=(8, 6))
    #plt.scatter(true_labels, error, color='blue', alpha=0.5)
    #plt.xlabel('Intended Values')
    #plt.ylabel('Prediction Error')
    #plt.title('Error vs. Intended Values')
    #plt.grid(True)
    #plt.savefig(f'{columnName}_error_plot.png')  # Change 'error_plot.png' to your desired filename and file format

    ##plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the plot as an image file
    plt.savefig('model_loss_plot.png')

    # Close the plot to free up resources
    plt.close()
