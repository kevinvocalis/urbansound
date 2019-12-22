from keras import Sequential
from keras.layers import Conv2D,  Activation, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D
from keras import optimizers
from subprocess import call
call("pip install tqdm".split(" "))
from tqdm import tqdm
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import argparse
import json
import os
import tensorflow as tf

def keras_model_fn(hyperparameters):
    
    HEIGHT = 128
    WIDTH = 251
    DEPTH = 1
    NUM_CLASSES = hyperparameters['num_classes']
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model will be transformed into a TensorFlow Estimator before training and it will be saved in a 
    TensorFlow Serving SavedModel at the end of training.

    Args:
        hyperparameters: The hyperparameters passed to the SageMaker TrainingJob that runs your TensorFlow 
                         training script.
    Returns: A compiled Keras model
    """
    model = Sequential()
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["mean_squared_error", "accuracy"])


    return model


def fetch_data_from_folds(descriptor, root_directory, folds):
    SHAPE = (128, 251)
    x = []
    y = []
    files_in_fold = descriptor[descriptor.fold.isin(folds)]
    print(f"There are {files_in_fold.shape[0]} files in this fold")
#     for index, row in tqdm(files_in_fold.iterrows(), total = files_in_fold.shape[0]):
    for index, row in files_in_fold.iterrows():
        filename = os.path.join(root_directory, f"fold{row.fold}", row['slice_file_name'].replace('.wav', '.npy')) 
        try: 
            image_raw = np.load(filename)
            if image_raw.shape[0] < SHAPE[0] or image_raw.shape[1] < SHAPE[1]:
                image_padded = np.zeros(SHAPE)
                image_padded[:image_raw.shape[0],:image_raw.shape[1]] = image_raw
            else:
                image_padded = image_raw[:SHAPE[0],:SHAPE[1]]

            x.append(image_padded)
            y.append(row['class'])
        except: 
            print(f"Failed to load or process: {filename}")
            
    return np.array(x), np.array(y)


#Create 10 folds from the 10 fold directories. We'll use these a bit later

def create_folds_from_folders():
    fold = list(range(1,11))

    train_folds = []
    validate_folds = []

    for i in range (0,10):
        train_folds.append(fold.copy())
        train_folds[i].remove(i+1)
        validate_folds.append(i+1)
        
    return train_folds, validate_folds



def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--data_dir', type=str, default='UrbanSound8K')
    parser.add_argument('--epochs', type=int, default=1)
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    print("Passed args")
        
    train_folds, validate_folds = create_folds_from_folders()
    
    root_dir = args.data_dir
    data_dir = os.path.join(root_dir, 'spectogram')
    meta_dir = os.path.join(root_dir, 'metadata')
    descriptor = pd.read_csv(os.path.join(meta_dir, "UrbanSound8K.csv"))
    
    train_auc_arr = []
    validate_auc_arr = []
    train_acc_arr = []
    validate_acc_arr = []
    
    for i in range(10):
        
        print("Building test and train sets")
        x_train, y_train = fetch_data_from_folds(descriptor, data_dir , list(train_folds[i]))
        x_validate, y_validate = fetch_data_from_folds(descriptor, data_dir, list([validate_folds[i]]))

        enc = OneHotEncoder(sparse = False)
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], x_train.shape[2], 1)
        y_train = y_train.reshape(-1,1)

        x_validate = x_validate.reshape(x_validate.shape[0],x_validate.shape[1], x_validate.shape[2], 1)
        y_validate = y_validate.reshape(-1,1)

        y_train = enc.fit_transform(y_train)
        y_validate = enc.transform(y_validate)

        hyperparameters = {'num_classes': len(descriptor['class'].unique())}

        model = keras_model_fn(hyperparameters)

        model.fit(x_train, y_train, epochs = args.epochs, verbose = 1)
        print("Done training")
        print("Now evaluating model")
        train_preds = model.predict(x_train)
        train_auc = roc_auc_score(y_train, train_preds)
        train_acc = accuracy_score(y_train,(np.array(train_preds) > 0.5).astype(np.int_) )
        print('Train AUC:', train_auc)
        print('Train ACC:', train_acc)
        
        validate_preds = model.predict(x_validate)
        validate_auc = roc_auc_score(y_validate, validate_preds)
        validate_acc = accuracy_score(y_validate,(np.array(validate_preds) > 0.5).astype(np.int_) )

        print('Validation AUC:', validate_auc)
        print('Validation ACC:', validate_acc)
            
        train_auc_arr.append(train_auc)
        validate_auc_arr.append(validate_auc)
        train_acc_arr.append(train_acc)
        validate_acc_arr.append(validate_acc)

    print(f"Experiment Train AUC: {np.mean(train_auc_arr)}")
    print(f"Experiment Validation AUC: {np.mean(validate_auc_arr)}")
    print(f"Experiment Train ACC: {np.mean(train_acc_arr)}")
    print(f"Experiment Validation ACC: {np.mean(validate_acc_arr)}")

