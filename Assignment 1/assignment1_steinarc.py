import numpy as np
import matplotlib.pyplot as plt


def OLS(X, y):
    Xt = np.transpose(X)
    temp = np.dot(Xt, X)
    A = np.linalg.inv(temp)
    w = np.dot(A, ( np.dot(Xt, y)))
    
    return w


def Emse(X, y, w):
    wt = np.transpose(w)
    Xt = np.transpose(X)
    yt = np.transpose(y)
    Xwt = np.transpose(np.dot(X, w))
    
    a = np.dot(wt, np.dot(Xt, np.dot(X, w)))
    b = 2 * (np.dot(Xwt, y))
    c = np.dot(yt, y)
    
    return a - b + c


#Exercise 2.2
print("Exercise 2.2")
raw_train_data = np.genfromtxt('dataset/regression/train_2d_reg_data.csv', delimiter=',')
y_train = np.transpose(np.array([raw_train_data[:,2][1:]]))
X_train = raw_train_data[:,0:2][1:]

raw_test_data = np.genfromtxt('dataset/regression/test_2d_reg_data.csv', delimiter=',')
y_test = np.transpose(np.array([raw_test_data[:,2][1:]]))
X_test = raw_test_data[:,0:2][1:]


w = OLS(X_train, y_train)
print("Weights")
print(w)

error_train = Emse(X_train, y_train, w)
print("Error for training data: {}".format(error_train))

error_test = Emse(X_test, y_test, w)
print("Error for test data: {}".format(error_test))
print("It seems the error is smaller for the test data than for the training data, weird.\nThen I will just assume that the training data was harder to classify than the test data, or maybe there's some error in my code. The error value of 3.29 is large, so no the model does not generalize well.")

#Exercise 2.3
print("Exercise 2.3")
raw_train_data = np.genfromtxt('dataset/regression/train_1d_reg_data.csv', delimiter=',')
x_train = np.transpose(np.array([raw_train_data[:,0][1:]]))
y_train = np.transpose(np.array([raw_train_data[:,1][1:]]))

raw_test_data = np.genfromtxt('dataset/regression/test_1d_reg_data.csv', delimiter=',')
x_test = np.transpose(np.array([raw_test_data[:,0][1:]]))
y_test = np.transpose(np.array([raw_test_data[:,1][1:]]))

w = OLS(x_train, y_train)
print("Weights")
print(w)

x = [i*0.1 for i in range(0, 10)]
y = [i * w[0,0] for i in x]

plt.scatter(x_train, y_train)
plt.plot(x,y, 'r')
plt.title("On training data")
plt.show()

plt.scatter(x_test, y_test)
plt.plot(x,y, 'r')
plt.title("On test data")
plt.show()

print("The line seems to fit the data quite well")

error_train = Emse(x_train, y_train, w)
print("Error for training data: {}".format(error_train))

error_test = Emse(x_test, y_test, w)
print("Error for test data: {}".format(error_test))

