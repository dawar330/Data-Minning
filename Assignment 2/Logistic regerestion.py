import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data
TrainingSet1 = pd.read_csv("data.csv")
Y = TrainingSet1["Output"].to_numpy()

TrainingSet1 = TrainingSet1.drop(columns="Output")
X = TrainingSet1.to_numpy()
X = X.reshape(401, 5000)

Thetas = np.zeros(401)
Thetas = Thetas.reshape(401, 1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hx(Thetas, Input_features):
    h = np.dot(Thetas.T, X)
    return sigmoid(h)


def derivative(Thetas, Input_features, Output):
    y_pred = hx(Thetas, X)
    diff = y_pred - Output
    g = np.dot(X, diff.T)
    return g


def accuracy(Thetas, Input_features, Output):
    total = 0
    correct = 0
    y_pred = hx(Thetas, Input_features)
    for i in range(5000):
        if y_pred[0, i] > 0.5 and Output[i] == 1:
            correct += 1

        if y_pred[0, i] < 0.5 and Output[i] == 0:
            correct += 1

        total += 1
    print("Accuracy: ")
    print(correct / total * 100)
    return y_pred



def cost(Thetas, Input_features, Output):
    y_pred = hx(Thetas, X)
    cost = -1 * sum((Output * (np.log(y_pred)))) + sum((1 - Output) * (np.log(1 - y_pred)))
    cost = sum(cost)
    # cost += -(np.sum((yi * (np.log(h(xi, theta)))) + ((1 - yi) * (np.log(1 - (h(xi, theta)))))))
    return cost / 5000


def Logistic(Thetas_New, Thetas_Old, Learning_rate, Iterations, classofinterest, Output):
    Alpha = Learning_rate
    iterations = Iterations
    output = Output
    for item in range(5000):
        if output[item] == classofinterest:
            output[item] = 1
        else:
            output[item] = 0

    cost1 = -1
    for it in range(iterations):

        dr = derivative(Thetas_Old, X, output)

        Thetas_New = Thetas_Old - (Alpha * dr)

        cost2 = cost(Thetas_New, X, output)
        Thetas_Old = Thetas_New
        if (cost1 - cost2) > 10 ^ -3:
            plt.figure("COST VS Iteration")
            plt.xlabel('Iteration')
            plt.ylabel('Cost Function')
            plt.scatter(it, cost2)
            cost1 = cost2
        else:
            print("Cost increased")
            plt.show()

            break
    accuracy(Thetas_New, X, output)
    print("Cost: ", cost2)
    plt.show()
    return Thetas_New


T = np.empty((401, 10))
T = T.reshape(10, 401)

print("For Class ", 0)
T[0] = Logistic(Thetas, Thetas, 0.00059, 100, 0, Y).T
print("For Class ", 1)
T[1] = Logistic(Thetas, Thetas, 0.00059, 100, 1, Y).T
print("For Class ", 2)
T[2] = Logistic(Thetas, Thetas, 0.00059, 100, 2, Y).T
print("For Class ", 3)
T[3] = Logistic(Thetas, Thetas, 0.00059, 100, 3, Y).T
print("For Class ", 4)
T[4] = Logistic(Thetas, Thetas, 0.00059, 100, 4, Y).T
print("For Class ", 5)
T[5] = Logistic(Thetas, Thetas, 0.00059, 100, 5, Y).T
print("For Class ", 6)
T[6] = Logistic(Thetas, Thetas, 0.00059, 100, 6, Y).T
print("For Class ", 7)
T[7] = Logistic(Thetas, Thetas, 0.00059, 100, 7, Y).T
print("For Class ", 8)
T[8] = Logistic(Thetas, Thetas, 0.00059, 100, 8, Y).T
print("For Class ", 9)
T[9] = Logistic(Thetas, Thetas, 0.00059, 100, 9, Y).T



# OneVsAll
print("One VS ALL")
total = 0
correct = 0
y = sigmoid(np.dot(T, X)).T
y = y.reshape(5000, 10)
print(y.shape)
for col in range(10):
    for row in range(5000):
        a = y[row]
        a = np.argmin(a)
        print("iteration ", row , a , Y[row] )
        if a == Y[row]:
            correct += 1
        total += 1

print(correct/total*100)
print(total)
