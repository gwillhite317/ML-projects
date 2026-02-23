import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report



titanic_data = pd.read_csv("C:/Users/mowma/Downloads/train.csv")
label = LabelEncoder()
titanic_data['Sex'] = label.fit_transform(titanic_data['Sex'])
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
features = titanic_data[["Pclass", "Sex", "Age", "SibSp", "Fare"]]
target = titanic_data[["Survived"]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size =.3, random_state = 42)

X_train, X_temp, y_train, y_temp = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

y_train = y_train.squeeze()
y_dev = y_dev.squeeze()
y_test = y_test.squeeze()


print("Train:", X_train.shape, y_train.shape)
print("Dev:", X_dev.shape, y_dev.shape)
print("Test:", X_test.shape, y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train,y_train.values.ravel())
log_predictions = log_model.predict(X_dev)
print(classification_report(y_dev, log_predictions))

log_tuned = LogisticRegression(C=.5, penalty='l2',solver = 'lbfgs',max_iter=400)
log_tuned.fit(X_train,y_train.values.ravel())
log_tuned_predictions = log_tuned.predict(X_dev)
print(classification_report(y_dev, log_tuned_predictions))

#vectorized knn model based off of some things I learned in data mining, uses euclidean distances and calculates sse for each dev.train pair
#results in the distance from the ith dev sample to jth training sample
def knn_predict(X_train, y_train, X_dev, k=3):
    dists = np.sqrt(((X_dev[:, np.newaxis] - X_train)**2).sum(axis=2))
    #sorts the distances, taking indices of k nearest neighbors
    k_idx = np.argsort(dists, axis=1)[:, :k]
    k_labels = y_train.to_numpy()[k_idx]
    #uses indices to get label of the nearest neuighbor
    preds = (k_labels.mean(axis=1) >= 0.5).astype(int)
    #take mean of neighbor labels and return mean
    return preds

knn_precits = knn_predict(X_train,y_train,X_dev,k=3)
print(classification_report(y_dev,knn_precits))

from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

dummy_1 = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_1.fit(X_train, y_train)
y_pred_mf = dummy_1.predict(X_dev)
print("most frequent")
print(classification_report(y_dev, y_pred_mf))

dummy_2 = DummyClassifier(strategy="stratified", random_state=42)
dummy_2.fit(X_train, y_train)
y_pred_strat = dummy_2.predict(X_dev)
print("stratified")
print(classification_report(y_dev, y_pred_strat))


log_best_pred = log_tuned.predict(X_test)
print("tuned log model")
print(classification_report(y_test, log_best_pred))

knn_test_pred = knn_predict(X_train, y_train, X_test)
print("best knn against test")
print(classification_report(y_test, knn_test_pred))

best_dummy = DummyClassifier(strategy="stratified")
best_dummy.fit(X_train, y_train)
best_dummy_pred = best_dummy.predict(X_test)
print("stratified dummy performance against test")
print(classification_report(y_test, best_dummy_pred))