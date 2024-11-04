import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)  #
x = Activation("relu")(x)
x = Dropout(0.3)(x)

x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.3)(x)

x = Dense(1, activation="sigmoid")(x)

for layer in base_model.layers[-15:]:
    layer.trainable = True

initial_learning_rate = 1e-5

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.95,
    staircase=True,
)

model = Model(inputs=base_model.input, outputs=x)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.2,
    fill_mode="nearest",
    dtype=np.float32,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    dtype=np.float32,
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    dtype=np.float32,
)

# Load the data
train_generator = train_datagen.flow_from_directory(
    "/home/ubuntu/chest_xray/train",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    "/home/ubuntu/chest_xray/val",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=False,
)

test_generator = test_datagen.flow_from_directory(
    "/home/ubuntu/chest_xray/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False,
)


# Create custom generator wrapper to match expected structure
def generator_wrapper(generator):
    while True:
        x, y = next(generator)  # Use next() directly on the generator
        yield (x, y, np.ones(y.shape))


# Wrap your generators
train_generator_wrapped = generator_wrapper(train_generator)
val_generator_wrapped = generator_wrapper(val_generator)


# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

try:
    # Try to get one batch and see what happens
    batch_x, batch_y = next(train_generator)
    print("Successfully got a batch!")

except Exception as e:
    print("Error type:", type(e))
    print("Error message:", str(e))

# Check validation generator
print("Validation samples:", val_generator.samples)
print("Validation batch size:", val_generator.batch_size)

# Adjust validation batch size if necessary
# Ensure that the batch size is <= number of validation samples
val_generator.batch_size = min(val_generator.batch_size, val_generator.samples)

# Calculate validation steps
val_steps = max(1, val_generator.samples // val_generator.batch_size)

try:
    history = model.fit(
        train_generator_wrapped,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator_wrapped,
        validation_steps=val_steps,
        epochs=15,
        verbose=1,
        callbacks=callbacks,
    )
except Exception as e:
    print("Error type:", type(e))
    print("Error message:", str(e))
    # Additional debugging information
    print("Steps per epoch:", train_generator.samples // train_generator.batch_size)
    print("Validation steps:", val_steps)

model.save('/home/ubuntu/models/pneumonia_model.keras')

test_dir = "/home/ubuntu/chest_xray/test"
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png")):
            test_image_path = os.path.join(root, file)
            # Load and preprocess the image
            img = tf.keras.preprocessing.image.load_img(
                test_image_path, target_size=(224, 224)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make the prediction
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])
            result = "Pneumonia" if confidence > 0.5 else "Normal"

            # Print the results
            print(f"Image: {test_image_path}")
            print(f"Prediction: {result} (confidence: {confidence:.2%})")
            print("---")