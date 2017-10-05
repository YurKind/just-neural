from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255
Y_test = np_utils.to_categorical(y_test, 10)
Y_train = np_utils.to_categorical(y_train, 10)

json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("mnist_model.h5")

loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("Точность модели на тестовых данных: %.2f%%" % (scores[1] * 100))
