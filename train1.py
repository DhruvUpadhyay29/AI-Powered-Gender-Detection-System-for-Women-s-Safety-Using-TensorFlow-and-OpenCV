import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN custom operations

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import glob

# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (224, 224, 3)  # Increase dimensions to capture the entire head

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'C:\Files\dataset' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labeling the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))  # Resize to new dimensions
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]  # C:\Files\gender_dataset_face\woman\face_1162.jpg
    label = 1 if label == "woman" else 0
    labels.append(label)

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2)  # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)

# augmenting dataset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

from tensorflow.keras.layers import Input

def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # Use Input layer
    model = Sequential([
        Input(shape=inputShape),
        Conv2D(32, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),

        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),

        Conv2D(128, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(1024),
        Activation("relu"),
        BatchNormalization(),
        Dropout(0.5),

        Dense(classes),
        Activation("sigmoid")
    ])

    return model


# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# use a learning rate schedule instead of decay
opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# save the model to disk
model.save('gender_detection_head.h5')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')
