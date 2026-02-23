import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1
class rosenblatt_perceptron:
    def __init__(self, lr = .01, iterations =100):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias= None

    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def fit_perceptron(self, x, y):
        n_samples, n_features = x.shape
        self.bias=0
        self.weights = np.zeros(n_features)

        for i in range(self.iterations):
            for all_x, xi in enumerate(x):
                linear = np.dot(xi, self.weights) + self.bias
                y_predict = self.activation(linear)
                update = self.lr * (y[all_x] - y_predict)

                self.weights += update * xi
                self.bias +=update

    
    def predict(self, x):
        linear_activation = np.dot(x, self.weights) + self.bias
        y_predicted = self.activation(linear_activation)
        return y_predicted
    

# 1 linearly seperable dataset testing
x = np.array([
    [1,4],[2,5],[3,8],[3,9],[4,10],  
    [5,1],[6,2],[7,3],[8,4],[9,5]     
])
y = np.array([1,1,1,1,1, 0,0,0,0,0])


perceptron = rosenblatt_perceptron(lr=0.1, iterations=1000)
perceptron.fit_perceptron(x, y)

y_pred = perceptron.predict(x)
accuracy = np.mean(y_pred == y)
print("Predictions:", y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Weights:", perceptron.weights, "Bias:", perceptron.bias)

# 2 Non-linearly seperable dataset training
x_2= np.array([
    [1,4],[2,5],[3,8],[7,0],[3,9],   
    [1,5],[2,3],[3,6],[7,2],[3,10]    
])
y_2 = np.array([1,1,1,1,1, 0,0,0,0,0])

perceptron_2 = rosenblatt_perceptron(lr=.1, iterations = 1000)
perceptron_2.fit_perceptron(x_2,y_2)
y_pred_2 = perceptron_2.predict(x_2)
accuracy_2 = np.mean(y_pred_2==y)

print("Predictions:", y_pred_2)
print(f"Accuracy: {accuracy_2 * 100:.2f}%")
print("Weights:", perceptron.weights, "Bias:", perceptron.bias)


# 3 titanic data 
titanic_data = pd.read_csv("C:/Users/mowma/Downloads/train.csv")
def format(entry):
    if entry == "male":
        return 1
    else:
        return 0  
titanic_data['Sex'] = titanic_data['Sex'].apply(format)

features = titanic_data[["Pclass","Sex", "Age", "SibSp", "Parch", "Fare"]]
target = titanic_data[["Survived"]]
features['Age'] = features['Age'].fillna(features['Age'].mean())
features.info()

X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, test_size=0.3, random_state=42)
y_train = y_train.ravel()
y_train = np.where(y_train == 0, -1, 1)
y_test = y_test.ravel()
y_test = np.where(y_test == 0, -1, 1)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)

#adaline model from github
class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
       
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

adaline = AdalineGD(eta=.001, n_iter = 100)
adaline.fit(x_train_scaled, y_train)
predict_adaline = adaline.predict(x_train_scaled)
accuracy_adaline = np.mean(predict_adaline == y_train)

#4
print(f"Accuracy of adaline model with training data: {accuracy_adaline * 100:.2f}%")
print(f"Weights of model with training data: ", adaline.w_)
# most influential weights were sex, class,sibsp in both models
adaline_2 = AdalineGD(eta=.001, n_iter = 100)
adaline_2.fit(x_test_scaled, y_test)
predict_adaline_2 = adaline_2.predict(x_test_scaled)
accuracy_adaline = np.mean(predict_adaline_2 ==y_test)

print(f"Accuracy of model on test data: {accuracy_adaline *100 :.2f}%")
print("Weights of model with test data: ", adaline_2.w_)



#5
values, counts = np.unique(y_train, return_counts=True)
majority_class = values[counts.argmax()]  
y_pred_majority = np.full_like(y_test, majority_class)
acc_majority = np.mean(y_pred_majority == y_test)
y_pred_random = np.random.choice(values, size=len(y_test))
acc_random = np.mean(y_pred_random == y_test)
print("Majority baseline accuracy:", acc_majority)
print("Random baseline accuracy:", acc_random)