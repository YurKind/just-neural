import numpy
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

json_file = open("cifar10_model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cifar10_model.h5")

loaded_model.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

classes = ['самолёт', 'автомобиль', 'птица', 'кот', 'олень',
           'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(32, 32))
plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)

prediction = loaded_model.predict(x)
prediction = numpy.argmax(prediction, axis=1)
print(classes[prediction[0]])