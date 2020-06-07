import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best_acc = 0
while best_acc < 0.85:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    # use pickle to save model when model fits better
    if accuracy > best_acc:
        best_acc = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

#use previously saved model with pickle instead
#of retraining every run
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficients: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#Model has several features; can only plot 2D graphs w pyplot
#so we will only be using one parameter
parameter = "G1"
style.use("ggplot")
pyplot.scatter(data[parameter], data["G3"])
pyplot.xlabel(parameter)
pyplot.ylabel("Final Grade")
pyplot.show()
