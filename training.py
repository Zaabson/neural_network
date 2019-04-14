import numpy as np
import pandas
from neural_network import ConvolutionalNeuralNetwork

def flower_name_to_list(flower):

    if flower == "Iris-setosa":
        return [1, 0, 0]
    if flower == "Iris-versicolor":
        return [0, 1, 0]
    if flower == "Iris-virginica":
        return [0, 0, 1]


iris_df = pandas.read_csv('iris_dataset.txt', header=None)
iris_df = iris_df.sample(frac=1).reset_index(drop=True)
train_x, test_x, train_y, test_y = iris_df.loc[:109,:3], iris_df.loc[110:,:3], iris_df.loc[:109, 4], iris_df.loc[110:, 4]
train_x, test_x = train_x.values, test_x.values
train_y = np.array([flower_name_to_list(flower_name) for flower_name in train_y])
test_y = np.array([flower_name_to_list(flower_name) for flower_name in test_y])

# creates NN with two hidden layers and sigmoids as activation functions
NN = ConvolutionalNeuralNetwork((4, 4, 4, 3), ('none', 'sigmoid', 'sigmoid', 'sigmoid'))

print("prior to training: ")
NN.evaluate(test_x, test_y)

NN.learn_RMSprop(train_x, train_y, epochs=100)
print("after 100 epochs: ")
NN.evaluate(test_x, test_y)

NN.learn_RMSprop(train_x, train_y, epochs=100)
print("after 200 epochs: ")
NN.evaluate(test_x, test_y)
