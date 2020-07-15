#Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = data.load_data()

#Interpret dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Preprocessing
train_images=train_images/255.0

test_images=test_images/255.0

#Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10)
])

model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])

model.fit(train_images,train_labels, epochs=10)

#Get accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels)

print('\nTest accuracy:', test_acc)

#Prediction
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
for x in range(len(test_images)):
    print("Actual: "+class_names[test_labels[x]]+"\tPrediction: "+class_names[np.argmax(predictions[x])])
