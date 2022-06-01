import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from evaluation.evaluation import *
from tensorflow.keras.utils import to_categorical
from models.model import *
import pickle
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
from tqdm import tqdm

random.seed(89)
seed(25)
tf.random.set_seed(40)

EPOCHS = 30
INPUT_SHAPE = (200, 200, 3)
BATCH_SIZE = 64
VAL_SPLIT = 0.2
IMG_BASE_PATH = "C:/Users/Administrator/Desktop/datasets/UTKface_Aligned_cropped/UTKFace/UTKFace/"
TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"
VAL_DATA_PATH = "../data/val.csv"
BEST_MODEL_PATH = "../trained_models/"
TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []
VAL_DATA = []
VAL_LABELS = []

"""
Male(0), Female(1) 
"""
# LABELS = [0, 1, 2, 3, 4]
# LABELS = ["Male", "Female"]
# NUM_CLASSES = len(LABELS)

print("Loading training data")
# Load the data and labels
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
val_df = pd.read_csv(VAL_DATA_PATH)

train_imgs = train_df["names"].tolist()
train_labels = train_df["age"].tolist()

test_imgs = test_df["names"].tolist()
test_labels = test_df["age"].tolist()

val_imgs = val_df["names"].tolist()
val_labels = val_df["age"].tolist()

for ti in tqdm(train_imgs):
    img_path = IMG_BASE_PATH + ti
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    TRAIN_DATA.append(img)

for vi in tqdm(val_imgs):
    img_path = IMG_BASE_PATH + vi
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    VAL_DATA.append(img)

for tt in tqdm(test_imgs):
    img_path = IMG_BASE_PATH + tt
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    TEST_DATA.append(img)

# Normalize data
print("Converting data")
TRAIN_DATA = np.array(TRAIN_DATA)
VAL_DATA = np.array(VAL_DATA)
TEST_DATA = np.array(TEST_DATA)
TEST_DATA = TEST_DATA / 255.

TRAIN_DATA = TRAIN_DATA.astype('float32')
VAL_DATA = VAL_DATA.astype('float32')
TEST_DATA = TEST_DATA.astype('float32')

# one-hot encode labels
print("Scale ages")
scaler = MinMaxScaler(feature_range=(0, 1))
train_labels = scaler.fit_transform(np.asarray(train_labels).reshape(-1, 1))
test_labels = scaler.fit_transform(np.asarray(test_labels).reshape(-1, 1))

TRAIN_LABELS = to_categorical(train_labels)
VAL_LABELS = to_categorical(val_labels)
TEST_LABELS = to_categorical(test_labels)

# data augmentation
print("Augmenting data")
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
)

# test_datagen = ImageDataGenerator(
#     rescale=1. / 255,
# )

TRAIN_DATA = np.reshape(TRAIN_DATA, (len(TRAIN_DATA), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
VAL_DATA = np.reshape(VAL_DATA, (len(VAL_DATA), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
TEST_DATA = np.reshape(TEST_DATA, (len(TEST_DATA), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))

train_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE)
val_gen = val_datagen.flow(VAL_DATA, VAL_LABELS, batch_size=BATCH_SIZE)
# test_gen = test_datagen.flow(TEST_DATA)


# setup callbacks
print("Setting up callbacks")
file_ext = ".h5"
name = "age"
callbacks = create_callbacks_reg()  # BEST_MODEL_PATH + name + file_ext, "loss", "min", 5)

# build model
print("Building model")
input_tensor = Input(shape=INPUT_SHAPE)
base = base_net(INPUT_SHAPE)
head = age_head()
net = build_model(base, head)
opt = Adam(learning_rate=0.001)

net.compile(optimizer=opt, loss="mse", metrics=['mse'])
net.summary()
#
# train model
print("Training model")
history = net.fit(train_gen,
                  validation_data=val_gen,
                  callbacks=callbacks,
                  epochs=EPOCHS)

# evaluate model
print("Evaluating model")
acc = net.evaluate(TEST_DATA, TEST_LABELS, batch_size=BATCH_SIZE)
print(acc)
# acc2 = net.evaluate(test_gen)
# print(acc2)
# preds = net.predict(TEST_DATA, verbose=0)
# preds = np.argmax(preds, axis=1)
#
# model_loss_path = "../graphs/" + name + "_loss.png"
# model_acc_path = "../graphs/" + name + "_acc.png"
# model_cm_path = "../graphs/" + name + "_cm.png"
# model_metrics_path = "../results/" + name + "_metrics.txt"
#
# print("Saving results")
# plot_confusion_matrix(TEST_LABELS, preds, LABELS, name, model_cm_path)
# acc_loss_graphs_to_file(name, history, ['train', 'val'], 'upper left', model_loss_path, model_acc_path)
# metrics_to_file(name, model_metrics_path, TEST_LABELS, preds, LABELS, acc)
#
# print("Done!")
