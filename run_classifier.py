from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from fema import FEMaClassifier



data = datasets.load_iris()
X = data.data
y = data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = FEMaClassifier()
# print(clf.score(X_train, y_train))

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'Accuracy = {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))