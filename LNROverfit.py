import numpy as np
import sklearn
import sklearn.linear_model

mean = 0
sigma = 1


a = 3
b = -5
c = 7

def f(x):
	epsilon = np.random.normal(mean, sigma)
    return a * x[0] + b * x[1]  + c * x[2] + epsilon

X = [(np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(1000)]
X_test = [(np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(50)]
y = [(f(_)) for _ in X]
y_test = [(f(_)) for _ in X_test]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)

model.fit(X, y)

print("a, b, c , epsilon = ", a, b, c, epsilon)
print("Model: ", model.coef_)

print(np.mean(np.abs(model.predict(X_test) - y_test)))


#Overfit:
X_overfit = [(x1, x2, x3, np.random.uniform(-100, -4), np.random.uniform(217, 355)) for x1, x2, x3 in X]
X_test_2 = [(x1, x2, x3, np.random.uniform(-100, -4), np.random.uniform(217, 355)) for x1, x2, x3 in X_test]
model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X_overfit, y)

print("Overfit: ", model.coef_)
print(np.mean(np.abs(model.predict(X_test_2) - y_test)))

#Undefit
X_underfit = [(x1, x3) for x1, x2, x3, in X]
X_test_3 = [(x1, x3) for x1, x2, x3 in X_test]
model = sklearn.linear_model.LinearRegression(fit_intercept=False)

model.fit(X_underfit, y)
print("Under fit:", model.coef_)
print(np.mean(np.abs(model.predict(X_test_3) - y_test)))