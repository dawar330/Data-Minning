import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


# Read data
dataset1 = pd.read_csv("data1.txt")
dataset2 = pd.read_csv("data2.txt")
X = dataset1['Population'].to_numpy()
Y = dataset1['Price'].to_numpy()

X2 = dataset2["Size"].to_numpy()
test = dataset2[["Size", "No.Bedrooms"]].to_numpy()
Y2 = dataset2['No.Bedrooms'].to_numpy()
Z2 = dataset2['Price'].to_numpy()

# Reshape and normalize data
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
print(X)
X = preprocessing.minmax_scale(X)
Y = preprocessing.minmax_scale(Y)

X2 = X2.reshape(-1, 1)
Y2 = Y2.reshape(-1, 1)
Z2 = Z2.reshape(-1, 1)

X2 = preprocessing.minmax_scale(X2)
Y2 = preprocessing.minmax_scale(Y2)
Z2 = preprocessing.minmax_scale(Z2)
test = preprocessing.minmax_scale(test)
model = LinearRegression()
model.fit(X, Y)


#
def gradient_descent_single_feature(x, y):
    Theta0 = Theta1 = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.91
    pre_cost = 0.17
    for i in range(iterations):

        y_predicted = Theta0  + Theta1 * x
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        Theta0 = Theta0 - learning_rate * bd
        Theta1 = Theta1 - learning_rate * md

        if (pre_cost - cost) > (10 ** -3):
            plt.figure("COST VS Iteration")
            plt.ylabel('Cost Function')
            plt.xlabel('Iteration')
            plt.scatter(i, cost)
            pre_cost = cost
        else:
            print("Hypothisis:")
            print("y = {}xo + {}x1, cost{} ".format(Theta0, Theta1, cost))
            print("Learning Rate:")
            print(learning_rate)
            plt.figure("DATA VS HYPOTHESIS")
            plt.ylabel('Price')
            plt.xlabel('Population')
            plt.scatter(X, Y)
            y = Theta1*x + Theta0
            plt.scatter(x, y, color="red")
            plt.show()
            break


def gradient_descent_Multiple_feature(x, y, z):
    Theta0 = Theta1 = Theta2 = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.67
    pre_cost = 0.08

    for i in range(iterations):

        z_predicted = Theta0  + Theta1 * x + Theta2 * y
        cost = (1 / (2 * n)) * sum([val ** 2 for val in (z - z_predicted)])
        md = -(2 / n) * sum(z - z_predicted)
        bd = -(2 / n) * sum(x * (z - z_predicted))
        cd = -(2 / n) * sum(y * (z - z_predicted))
        Theta0 = Theta0 - learning_rate * md
        Theta1 = Theta1 - learning_rate * bd
        Theta2 = Theta2 - learning_rate * cd

        if (pre_cost - cost) > (10 ** -3):
            plt.figure("COST VS Iteration Data 2")
            plt.ylabel('Cost Function')
            plt.xlabel('Iteration')
            plt.scatter(i, cost)

            pre_cost = cost
        else:
            print("Hypothisis:")
            print("z = {}xo + {}x1 + {}x2, cost{} ".format(Theta0, Theta1, Theta2, cost))
            print("Learning Rate:")
            print(learning_rate)
            plt.figure("DATA VS HYPOTHESIS Data 2")
            ax = plt.axes(projection='3d')
            ax.scatter3D(X2, Y2, Z2)
            z = Theta0 + Theta1*x + Theta2*y
            ax.scatter3D(x, y, z, color='red')
            ax.set_xlabel("Size")
            ax.set_ylabel("Bedrooms")
            ax.set_zlabel("Price")
            plt.show()
            break


gradient_descent_Multiple_feature(X2, Y2, Z2)
gradient_descent_single_feature(X, Y)
