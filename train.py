import we_helpers as wh
import data_helpers as dh
import model as m
import numpy as np
import time
import tensorflow as tf

print("Loading data")
x_data, y_data = dh.load_data_and_labels("./data/data.pos", "./data/data.neg")

for i in range(len(x_data)):
    x_data[i] = wh.sentence2matrix(x_data[i])

split_point = -1*int(0.1*float(len(x_data)))

np.random.seed(20)

shuffle_indices = np.random.permutation(np.arange(len(x_data)))
shuffled_x_data = np.array(x_data)[shuffle_indices]
shuffled_y_data = np.array(y_data)[shuffle_indices]

x_train = np.array(shuffled_x_data[:split_point])
x_test = np.array(shuffled_x_data[split_point:])
y_train = np.array(shuffled_y_data[:split_point])
y_test = np.array(shuffled_y_data[split_point:])

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
print("Loading data finished")

print("Get models")
model = m.get_model()

print("Training")
model.fit([x_train, x_train, x_train, x_train], y_train, batch_size=4, epochs=30)

print("\nEval")
score = model.evaluate([x_test, x_test, x_test, x_test], y_test, batch_size=4)
print("Loss:{}\nAcc:{}".format(score[0], score[1]))
print("")
model.save("./models/model{}-{}-{}".format(int(time.time()), int(score[0]*100), int(score[1]*100)))
