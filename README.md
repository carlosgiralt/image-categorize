# Image categorize

Project to train a model to categorize images, depending on the categories defined in your project structure.

**Example:** the training and test datasets have the same structure and contains two classes "class1" and "class2".

```
samples/dataset
    |-----class1/
    |       |
    |       |-----image1.jpg
    |       |-----image2.jpg
    |       |-----image3.jpg
    |       |-----...
    |
    |-----class2/
    |       |
    |       |-----image1.jpg
    |       |-----image2.jpg
    |       |-----image3.jpg
    |       |-----...
    |
    |-----...
```
The `train.py` script will train, validate and create a model based on the
`training` folder and the `model_name` variable which represents the dataset
and the model name.

The `categorize.py` script will load the corresponding model and it will
detect and classify the images in the `samples` folder.

---

## Run an example locally

### Create the enviroment:
```sh
virtualenv -p python3 ~/.virtualenvs/<virtualenv_name>
```

### Activate the environment:
```sh
source ~/.virtualenvs/<virtualenv_name>/bin/activate
```

### Install the project dependencies:
```sh
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Setup the training tests:
- Suppose your project will detect animals
- Create a folder inside the `training` directory. Example: `animal_detection`
- Put inside your new the pictures like in the following example:
  ```
  training/animal_detection
    |-----rabbit/
    |       |
    |       |-----image1.jpg
    |       |-----image2.jpg
    |       |-----image3.jpg
    |       |-----...
    |
    |-----horse/
    |       |
    |       |-----image1.jpg
    |       |-----image2.jpg
    |       |-----image3.jpg
    |       |-----...
    |
    |----
  ```
- Create a folder inside the `samples` directory. Example: `animal_detection`
- Put inside your new the pictures that you want to categorize.
    ```
    samples/animal_detection
        |-----unknown_001.jpg
        |-----unknown_002.jpg
        |-----unknown_003.jpg
    ```

### Train your model:
```sh
python train.py
```
This will create a the file `models/<project_name>`. Example: `models/animal_detection`.

### Categorize your examples:
```sh
python categorize.py
```
This will create a the file `out/<project_name>/results.txt`. Example: `models/animal_detection/results.txt`.

The results file will look like the following:
```
{'filename': 'unknown_001.jpg', 'prediction': [0.091228105], 'category': 'rabbit'}
{'filename': 'unknown_002.jpg', 'prediction': [0.49978378], 'category': 'horse'}
{'filename': 'unknown_003.jpg', 'prediction': [0.50978378], 'category': 'rabbit'}
```

## I hope this is useful for you. :)
