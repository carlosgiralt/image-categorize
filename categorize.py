import os
from itertools import chain

from keras.preprocessing import image
from keras.models import load_model

from train import BASE_DIR, model_name, train_generator

SAMPLES_DIR = BASE_DIR / f"samples/{model_name}"
OUTPUT_DIR = BASE_DIR / f"out/{model_name}"

# load the trained model
model_file = BASE_DIR / f"models/{model_name}.h5"

# get the category names from the training data
class_names = train_generator.class_indices
category_names = dict((v, k) for k, v in class_names.items())


def predict(filename, image_path):
    # load the trained model
    model = load_model(model_file)

    # preprocess the image
    img = image.image_utils.load_img(image_path, target_size=(150, 150))
    x = image.image_utils.img_to_array(img)
    x = x / 255.0
    x = x.reshape(-1, 150, 150, 3)

    # classify the image
    predictions = model.predict(x)
    prediction = list(chain.from_iterable(predictions))

    # get the highest probability category
    highest_prediction = prediction.index(max(prediction))
    category = category_names[highest_prediction]

    # print the classification result
    return dict(filename=filename, prediction=prediction, category=category)


# List all the images in the samples directory
IMG_EXT = ["jpg", "jpeg"]
images = []
for ext in IMG_EXT:
    images.extend(list(SAMPLES_DIR.glob(f"*.{ext}")))

results = []
for f in images:
    filename = f.name
    image_path = str(f.absolute())

    results.append(predict(filename, image_path))

# create output dir if does not exists
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# save the results in a file
with open(OUTPUT_DIR / "results.txt", "w+") as f:
    for line in results:
        f.write("\n" + str(line))
