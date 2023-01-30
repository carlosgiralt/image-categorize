from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop


BASE_DIR = Path(__file__).resolve(strict=True).parent


# specify the model name
model_name = "sample_model"

# specify the paths for the training and test datasets
TRAIN_DATA_DIR = BASE_DIR / f"training/{model_name}"

# set the batch size and number of epochs
batch_size = 32
epochs = 20

# create the data generator for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# create the data generator for the test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# generate the training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
)

# generate the test data
test_generator = test_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
)

# create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))

# compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(),
    metrics=["accuracy"],
)

# train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size,
)

# save the model
model_filepath = f"models/{model_name}.h5"
model.save(model_filepath)
