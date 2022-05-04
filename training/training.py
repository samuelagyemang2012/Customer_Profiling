import pandas as pd
import cv2
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from evaluation.evaluation import *
from tensorflow.keras.utils import to_categorical
from models.model import *
from numpy.random import seed
import random
import tensorflow as tf

random.seed(89)
seed(25)
tf.random.set_seed(40)

EPOCHS = 100
INPUT_SHAPE = (200, 200, 3)
BATCH_SIZE = 8
NUM_CLASSES = 5
VAL_SPLIT = 0.2
IMG_BASE_PATH = "C:/Users/Administrator/Desktop/datasets/UTKface_Aligned_cropped/UTKFace/UTKFace/"
TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"
BEST_MODEL_PATH = "../trained_models/"
TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []

"""
White(0), Black(1), Asian(2), Indian(3), Others(4) 
"""
LABELS = [0, 1, 2, 3, 4]

print("Loading training data")
# Load the data and labels
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

train_imgs = train_df["names"].tolist()
train_labels = train_df["race"].tolist()

test_imgs = test_df["names"].tolist()
test_labels = test_df["race"].tolist()

for ti in train_imgs:
    img_path = IMG_BASE_PATH + ti
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    TRAIN_DATA.append(img)

for tt in test_imgs:
    img_path = IMG_BASE_PATH + tt
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    TEST_DATA.append(img)

# Normalize data
print("Converting data")
TRAIN_DATA = np.array(TRAIN_DATA)
TEST_DATA = np.array(TEST_DATA)

TRAIN_DATA = TRAIN_DATA.astype('float32')
TEST_DATA = TEST_DATA.astype('float32')

# one-hot encode labels
print("One-hot encoding labels")
TRAIN_LABELS = to_categorical(train_labels)
TEST_LABELS = to_categorical(test_labels)

# data augmentation
print("Augmenting data")
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    validation_split=VAL_SPLIT,
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

TRAIN_DATA = np.reshape(TRAIN_DATA, (len(TRAIN_DATA), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
TEST_DATA = np.reshape(TEST_DATA, (len(TEST_DATA), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))

train_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='training')
val_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='validation')
test_gen = test_datagen.flow(TEST_DATA)

# setup callbacks
print("Setting up callbacks")
file_ext = ".h5"
name = "race"
callbacks = create_callbacks()  # BEST_MODEL_PATH + name + file_ext, "loss", "min", 5)

# build model
print("Building model")
input_tensor = Input(shape=INPUT_SHAPE)
# _resnet_base = resnet_50(input_tensor, INPUT_SHAPE, None, None)
# _fully_connected = fully_connected(NUM_CLASSES)
# model = build_model(_resnet_base, _fully_connected)
net = build_net(NUM_CLASSES, INPUT_SHAPE)

opts = [
    Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    Adam(learning_rate=0.0001),
]

# model = build_net(4)
net.compile(optimizer=opts[1], loss="categorical_crossentropy", metrics=['accuracy'])
net.summary()

# train model
print("Training model")
history = net.fit(train_gen,
                  validation_data=val_gen,
                  callbacks=callbacks,
                  # steps_per_epoch=len(TRAIN_DATA)/BATCH_SIZE,
                  epochs=EPOCHS)

# evaluate model
print("Evaluating model")
acc = net.evaluate(TEST_DATA, TEST_LABELS, batch_size=BATCH_SIZE)
preds = net.predict(TEST_DATA, verbose=0)
preds = np.argmax(preds, axis=1)
model_loss_path = "../graphs/" + name + "_loss.png"
model_acc_path = "../graphs/" + name + "_acc.png"
model_cm_path = "../graphs/" + name + "_cm.png"
plot_confusion_matrix(TEST_LABELS, preds, LABELS, name, model_cm_path)
acc_loss_graphs_to_file(name, history, ['train', 'val'], 'upper left', model_loss_path, model_acc_path)
model_metrics_path = "../results/" + name + "_metrics.txt"
metrics_to_file(name, model_metrics_path, TEST_LABELS, preds, LABELS, acc)

print("Done!")
