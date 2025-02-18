from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

import gc
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.trainers.data_adapters.py_dataset_adapter")


log_file_path = '../02-data/02-classifier/00-model/tensorflow_rendezvous_logs.txt'
sys.stderr = open(log_file_path, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only capture errors, to avoid excessive logs

train_dir = ('../02-data/02-classifier/train')
valid_dir = '../02-data/02-classifier/valid'
test_dir = '../02-data/02-classifier/test'

batch_size = 16
epochs = 20
img_height, img_width = 150,150

## 1. Transformations that will be applied to the sample images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

## Aply previous transformations to the train_dir
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(img_height, img_width),batch_size=batch_size,class_mode='binary')

## Same process for validation.
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)


## 2. What model are we gonna use? --> testing VGG16 pretrained
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

base_model.trainable = False  # Freeze the pre-trained layers

# Compile the model and free resources
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
gc.collect()


## 3. Create conditions to check every iteration and train
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint_callback = ModelCheckpoint('../02-data/02-classifier/00-model/best_image_classifier.keras', monitor='val_accuracy',save_best_only=True,mode='max',verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    callbacks=[checkpoint_callback,lr_scheduler,early_stopping]
)

# Evaluate the model on test data (if available)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"Test accuracy: {test_acc}")

# Evaluate the model on test data (using the best saved model)
best_model = load_model('../02-data/02-classifier/00-model/best_image_classifier.keras')
test_loss, test_acc = best_model.evaluate(test_generator, verbose=2)
print(f"Test accuracy (best model): {test_acc}")
model.save('../02-data/02-classifier/00-model/simple_image_classifier.keras')