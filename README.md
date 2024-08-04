# Chatbot with TensorFlow and NLTK

This repository contains the code for a simple chatbot implemented using TensorFlow and NLTK. The chatbot is trained on a set of intents defined in a JSON file. It can process user input, predict the intent, and provide an appropriate response.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Running the Chatbot](#running-the-chatbot)
- [Files](#files)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

This project demonstrates a basic chatbot using deep learning techniques. The bot is designed to understand user queries and provide responses based on pre-defined intents.

## Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK data**:

    The chatbot uses the NLTK library for tokenizing and lemmatizing text. You need to download the required NLTK data:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

## Training the Model

To train the model, run the training script:

```python
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '-', '.']

# Process each intent and pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Model training and saving completed")
```
## Running the Chatbot
To start the chatbot, run the following script:
```python
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Great! The bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
```
This script reads the intents from intents.json, processes the text, and trains a neural network model. The trained model is saved as chatbot_model.h5.
## Files

intents.json: JSON file containing the intents, patterns, and responses.
words.pkl: Pickle file containing the processed words.
classes.pkl: Pickle file containing the intent classes.
chatbot_model.h5: Trained model file.
training_script.py: Script to train the chatbot model.
chatbot_script.py: Script to run the chatbot.

## Dependencies

Dependencies
Python 3.x
TensorFlow
NLTK
NumPy
Keras
Pickle

Install the dependencies using:
```bash
pip install tensorflow nltk numpy keras
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Save this as `README.md` in your repository. Make sure to replace `your-username` and `your-repository-name` with your actual GitHub username and repository name. You might also want to add a `requirements.txt` file with the necessary dependencies for easier setup.

Here is an example of a `requirements.txt` file:

```txt
tensorflow
nltk
numpy
keras
pickle-mixin



