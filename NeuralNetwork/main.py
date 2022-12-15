##Authors: Karol Kuchnio s21912 and Michał Wadas s20495

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

"""
    * Wine quality * START
"""
print("* Wine quality *")
wineQuality_data = pd.read_csv('winequality-white.csv', sep=';',
                               names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                                      'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                      'alcohol', 'quality'])

wineQuality_train_data = wineQuality_data.copy()
wineQuality_train_label = wineQuality_data.pop('quality')
wineQuality_train_data = np.asarray(wineQuality_train_data).astype('float32')

normalize = tf.keras.layers.Normalization()
normalize.adapt(wineQuality_train_data)
wineQuality_model = tf.keras.Sequential([normalize, tf.keras.layers.Dense(128, activation='relu'),
                                         tf.keras.layers.Dense(64, activation='relu'),
                                         tf.keras.layers.Dense(41, activation='softmax')
                                         ])

wineQuality_model.compile(optimizer='adam',
                          loss='mean_squared_error',
                          metrics=['accuracy']
                          )
print(wineQuality_train_data)
wineQuality_model.fit(wineQuality_train_data, wineQuality_train_label, epochs=10)
"""
    * Wine quality * END
"""

"""
   * Confusion Matrix - CIFAR10 * START
"""
print("* Confusion Matrix - CIFAR10 *")
cifar10_data = tf.keras.datasets.cifar10
(c_train_data, c_train_label), (c_test_data, c_test_label) = cifar10_data.load_data()
c_train_data = c_train_data / 255
c_test_data = c_test_data / 255

cifar3_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(256, activation='relu'),
     tf.keras.layers.Dense(10, activation='softmax')
     ])
cifar3_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']
                     )
cifar3_model.fit(c_train_data, c_train_label, epochs=5)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_classes_prediction = np.argmax(cifar3_model.predict(c_test_data), axis=1)
confusion_matrix = tf.math.confusion_matrix(labels=c_test_label, predictions=image_classes_prediction).numpy()
confusion_matrix_norm = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
                                  decimals=2)
confusion_matrix_df = pd.DataFrame(confusion_matrix_norm, index=class_names, columns=class_names)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

print("* Confusion Matrix - CIFAR10 *  5 epochs ")
cifar10_data = tf.keras.datasets.cifar10
(c_train_data, c_train_label), (c_test_data, c_test_label) = cifar10_data.load_data()
c_train_data = c_train_data / 255
c_test_data = c_test_data / 255

bird_id = np.where(c_train_label.reshape(-1) == 2)
bird_data = c_train_data[bird_id]
bird_label = c_train_label[bird_id]

cat_id = np.where(c_train_label.reshape(-1) == 3)
cat_data = c_train_data[cat_id]
cat_label = c_train_label[cat_id]

deer_id = np.where(c_train_label.reshape(-1) == 4)
deer_data = c_train_data[deer_id]
deer_label = c_train_label[deer_id]

dog_id = np.where(c_train_label.reshape(-1) == 5)
dog_data = c_train_data[dog_id]
dog_label = c_train_label[dog_id]

frog_id = np.where(c_train_label.reshape(-1) == 6)
frog_data = c_train_data[frog_id]
frog_label = c_train_label[frog_id]

horse_id = np.where(c_train_label.reshape(-1) == 7)
horse_data = c_train_data[horse_id]
horse_label = c_train_label[horse_id]

animals_train_data = np.concatenate((bird_data, cat_data, deer_data, dog_data, frog_data, horse_data))
animals_train_label = np.concatenate((bird_label, cat_label, deer_label, dog_label, frog_label, horse_label)).reshape(
    -1, 1)
animals_train_label[animals_train_label == 2] = 0
animals_train_label[animals_train_label == 3] = 1
animals_train_label[animals_train_label == 4] = 2
animals_train_label[animals_train_label == 5] = 3
animals_train_label[animals_train_label == 6] = 4
animals_train_label[animals_train_label == 7] = 5

cifar3_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(256, activation='relu'),
     tf.keras.layers.Dense(6, activation='softmax')
     ])
cifar3_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']
                     )
cifar3_model.fit(animals_train_data, animals_train_label, epochs=5)

print("* Confusion Matrix - CIFAR10 *  10 epochs ")
cifar10_data2 = tf.keras.datasets.cifar10
(c2_train_data, c2_train_label), (c2_test_data, c2_test_label) = cifar10_data2.load_data()
c2_train_data = c2_train_data / 255
c2_test_data = c2_test_data / 255

bird2_index = np.where(c_train_label.reshape(-1) == 2)
bird2_data = c2_train_data[bird2_index]
bird2_label = c2_train_label[bird2_index]

cat2_index = np.where(c2_train_label.reshape(-1) == 3)
cat2_data = c2_train_data[cat2_index]
cat2_label = c2_train_label[cat2_index]

deer2_index = np.where(c_train_label.reshape(-1) == 4)
deer2_data = c2_train_data[deer2_index]
deer2_label = c2_train_label[deer2_index]

dog2_index = np.where(c2_train_label.reshape(-1) == 5)
dog2_data = c2_train_data[dog2_index]
dog2_label = c2_train_label[dog2_index]

frog2_index = np.where(c2_train_label.reshape(-1) == 6)
frog2_data = c2_train_data[frog2_index]
frog2_label = c2_train_label[frog2_index]

horse2_index = np.where(c2_train_label.reshape(-1) == 7)
horse2_data = c2_train_data[horse2_index]
horse2_label = c2_train_label[horse2_index]

animals2_train_data = np.concatenate((bird2_data, cat2_data, deer2_data, dog2_data, frog2_data, horse2_data))
animals2_train_label = np.concatenate(
    (bird2_label, cat2_label, deer2_label, dog2_label, frog2_label, horse2_label)).reshape(-1, 1)
animals2_train_label[animals2_train_label == 2] = 0
animals2_train_label[animals2_train_label == 3] = 1
animals2_train_label[animals2_train_label == 4] = 2
animals2_train_label[animals2_train_label == 5] = 3
animals2_train_label[animals2_train_label == 6] = 4
animals2_train_label[animals2_train_label == 7] = 5

cifar32_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(256, activation='relu'),
     tf.keras.layers.Dense(6, activation='softmax')
     ])
cifar32_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
cifar32_model.fit(animals2_train_data, animals2_train_label, epochs=10)

"""
   * Confusion Matrix - CIFAR10 * END
"""

"""
    * 10 Types of Clothes * START
"""
print("* 10 Types of Clothes * ")
data_clothes = tf.keras.datasets.fashion_mnist
(training_data_clothes, clothes_train_label), (test_data_clothes, clothes_test_label) = data_clothes.load_data()
training_data_clothes = training_data_clothes / 255.0
test_data_clothes = test_data_clothes / 255.0

clothes_model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                     tf.keras.layers.Dense(128, activation='relu'),
                                     tf.keras.layers.Dense(10, activation='softmax')
                                     ])
clothes_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
clothes_model.fit(training_data_clothes, clothes_train_label, epochs=10)

"""
    * 10 Types of Clothes classifying * END
"""

"""
    * Breast Cancer (Haberman) * START
"""
print("* Breast Cancer (Haberman) * ")
haberman_data = pd.read_csv('breastCancerHaberman.csv',
                            names=['age', 'Patient`s year of operation', 'Number of positive axillary nodes detected',
                                 'Survival status'])
haberman_train_data = haberman_data.copy()
haberman_train_label = haberman_data.pop('Survival status')
haberman_train_data = np.asarray(haberman_train_data).astype(np.float)

haberman_model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                      tf.keras.layers.Dense(1)
                                      ])
haberman_model.compile(optimizer='adam',
                       loss='mean_squared_error',
                       metrics=['accuracy']
                       )
haberman_model.fit(haberman_train_data, haberman_train_label, epochs=13)

"""
    * Breast Cancer (Haberman) * END
"""
