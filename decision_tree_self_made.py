import math
from sklearn.datasets import load_breast_cancer

def my_train_test_split(input_data, output_data, test_partion=0.5):
    input_length = len(input_data)
    output_length = len(output_data)
    if input_length != output_length:
        raise Exception("input_length != output_length")
    test_size = math.ceil(input_length * test_partion)
    train_size = input_length - test_size
    X_train = input_data[0:train_size]
    Y_train = output_data[0:train_size]
    X_test = input_data[train_size:]
    Y_test = output_data[train_size:]
    return X_train, X_test, Y_train, Y_test




if __name__ == "__main__":
    cancer = load_breast_cancer()
    print(cancer.keys())

    X_train, X_test, y_train, y_test = my_train_test_split(cancer.data, cancer.target, test_partion=1.0/5)