# pyFEMa: Python API for Finite Element Machine
pyFEMa is a [Finite Element Machine](https://github.com/danillorp/libFemClassifier) implemented using Scikit-Learn `sklearn.base.BaseEstimator`.

pyFEMa has two estimators:

- Finite Element Machine Classifier (FEMa) \
Estimator for Classification Task

- Finite Element Machine Regressor (FEMaR) \
Estimator for Regression Task

# Installation

## ~~Installation from binaries~~ (Pending)
~~Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pyfema.~~

```bash
pip install pyfema
```

## Installation from source
Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
---
Example of Classification Task.


```python
from fema import FEMaClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = datasets.load_iris()
X = data.data
y = data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = FEMaClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

# Reference
https://github.com/danillorp/libFemClassifier

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

# License
[MIT](https://choosealicense.com/licenses/mit/)