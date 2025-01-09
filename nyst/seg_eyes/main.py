import os
from glob import glob
import pickle

import keras
from sklearn.model_selection import train_test_split

from deeplab_mdl_def import DeeplabV3Plus
from utils import *

from deeplab_mdl_def import DynamicUpsample

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
import sys

# Add the 'code' directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 448
BATCH_SIZE = 16
NUM_CLASSES = 4
NUM_EPOCHS = 100
CALCULATE_CLASS_WEIGHTS = False
RELOAD_TRAINED_MODEL= False
DATA_DIR = "/repo/porri/Eyes Segmentation Dataset"

VAL_IMG_FRAC = 0.2
TEST_IMG_FRAC = 0.1
SUBSET = False
SUBSET_SIZE = 240
COLORMAP = patch_colors_bgr_01 = {
    "background": [0, 0, 0],  # BGR
    "pupil": [1, 0, 0],   # BGR 
    "eyes": [0, 0, 1],  # BGR 
    "iris": [0, 1, 0],  # BGR 
     
}
COLORMAP = {key: [color[2], color[1], color[0]] for key, color in COLORMAP.items()}

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
all_images = sorted(glob(os.path.join(DATA_DIR, "Images_f/*")))
all_masks = sorted(glob(os.path.join(DATA_DIR, "Masks_g_f/*")))
# Subset if necessary
if SUBSET:
    all_images = all_images[:SUBSET_SIZE]
    all_masks = all_masks[:SUBSET_SIZE]

# Get number of train, val and test images
NUM_VAL_TRAIN_IMAGES = int((VAL_IMG_FRAC + TEST_IMG_FRAC) * len(all_images))
NUM_VAL_IMAGES = int(VAL_IMG_FRAC * len(all_images))
NUM_TEST_IMAGES = NUM_VAL_TRAIN_IMAGES - NUM_VAL_IMAGES
NUM_TRAIN_IMAGES = len(all_images) - NUM_VAL_TRAIN_IMAGES

train_images, val_images, train_masks, val_masks = train_test_split(all_images, all_masks,
                                                                    test_size=NUM_VAL_TRAIN_IMAGES, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(val_images, val_masks,
                                                                  test_size=NUM_TEST_IMAGES, random_state=42)

print("Train Images: {} | expected: {}".format(len(train_images), NUM_TRAIN_IMAGES))
print("Train Masks: {} | expected: {}".format(len(train_masks), NUM_TRAIN_IMAGES))
print("Val Images: {} | expected: {}".format(len(val_images), NUM_VAL_IMAGES))
print("Val Masks: {} | expected: {}".format(len(val_masks), NUM_VAL_IMAGES))
print("Test Images: {} | expected: {}".format(len(test_images), NUM_TEST_IMAGES))
print("Test Masks: {} | expected: {}".format(len(test_masks), NUM_TEST_IMAGES))

train_dataset = data_generator(train_images,train_masks, BATCH_SIZE)
val_dataset = data_generator(val_images, val_masks, BATCH_SIZE)
test_dataset = data_generator(test_images, test_masks, BATCH_SIZE)
test_dataset_no_resize = data_generator(test_images, test_masks, BATCH_SIZE, augment_data=False, resize_image=False)

if CALCULATE_CLASS_WEIGHTS or not os.path.exists('class_weights.pkl'):
    print("Calculating class weights...")
    class_weights = read_masks_and_compute_weights(os.path.join(DATA_DIR, "Masks_g/"))
    with open('class_weights.pkl', 'wb') as pickle_file:
        pickle.dump(class_weights, pickle_file)
else:
    print("Reloading class weights pickle...")
    with open('class_weights.pkl', 'rb') as pickle_file:
        class_weights = pickle.load(pickle_file)

print("All done, class weights:")
print(class_weights)
#mdlname = "deeplabv3plus_face_segmentation_augmentation_class_weights_latest_fixConv.h5"
#mdlname = "eyes_seg.h5"
mdlname = "eyes_seg_threshold.h5"

if not RELOAD_TRAINED_MODEL or not os.path.exists(mdlname):
    print("Training model...")
    model = DeeplabV3Plus(num_classes=NUM_CLASSES)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = focal_loss_multiclass(alpha=0.25, gamma=2.0)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    print(model.summary())

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS, callbacks=[early_stopping],
                        class_weight=class_weights)

    model.save(mdlname)


    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.title("Training Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["val_loss"])
    plt.title("Validation Loss")
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["val_accuracy"])
    plt.title("Validation Accuracy")
    plt.ylabel("val_accuracy")
    plt.xlabel("epoch")
    plt.show()
else:
    print("Reloading model...")
    model = keras.models.load_model(mdlname, custom_objects={'DynamicUpsample': DynamicUpsample})

# test for different image sizes
print("Evaluating model on images with different sizes")
test_loss, test_accuracy = model.evaluate(test_dataset_no_resize)
print(f"Test loss (No resize): {test_loss}, Test accuracy: {test_accuracy}")

print("Evaluating model on images with uniform size (resize)")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
