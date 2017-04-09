from keras.models import load_model
import we_helpers as wh
import numpy as np

m_model = load_model('./models/model1490017738-54-71')
test_case = [
    "I don't like it, i hate it",
    "Amazing! It's a masterpiece",
    "What a boring film, i'll never watch it again",
    "illuminating if overly talky documentary",
    "interesting , but not compelling",
    "offers a breath of the fresh air of true sophistication"
]

x_test = []
for tc in test_case:
    x_test.append(wh.sentence2matrix(tc))

x_test = np.array(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

result = m_model.predict(x_test)
print(result)
