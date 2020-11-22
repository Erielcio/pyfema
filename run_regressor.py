from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
import math

from fema import FEMaRegressor

# House Pricing
data = datasets.load_boston()
X = data.data
y = data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regr = FEMaRegressor()
# print(regr.score(X_train, y_train))

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print(f'RMSE = {math.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'MAE = {mean_absolute_error(y_test, y_pred)}')
print(f'RMSLE = {math.sqrt(mean_squared_log_error(y_test, y_pred))}')
print(f'R2 = {r2_score(y_test, y_pred)}')