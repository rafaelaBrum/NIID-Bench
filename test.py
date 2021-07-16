import numpy as np

def test():
    X_train = np.load("data/generated/X_train.npy")
    X_test = np.load("data/generated/X_test.npy")
    y_train = np.load("data/generated/y_train.npy")
    y_test = np.load("data/generated/y_test.npy")
    print(X_train)

# np.save("data/generated/X_train.npy", X_train)
# np.save("data/generated/X_test.npy", X_test)
# np.save("data/generated/y_train.npy", y_train)
# np.save("data/generated/y_test.npy", y_test)
