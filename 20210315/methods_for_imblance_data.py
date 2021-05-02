from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
data = pd.read_csv("Churn_Modelling.csv")
print(data.info())

fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
fig.set_size_inches(14, 10)
sns.distplot(data.EstimatedSalary, color='#2D008E', ax=ax[0])
sns.distplot(data.CreditScore, color='#F9971A', ax=ax[1])
sns.distplot(data.Balance, color='#242852', ax=ax[2])
sns.distplot(data.Age, color='#242852', ax=ax[3])
# plt.show()
data2_churn_1 = data[data["Exited"] == 1]
data2_churn_0 = data[data["Exited"] == 0]
print(data2_churn_1.shape, data2_churn_0.shape)
data["Gender"] = np.where(data["Gender"] == 'Female', 0, 1)
x = data.drop(["Exited", "Geography", "Surname"], axis=1)
y = data["Exited"]
y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape, x_test.shape, y_train.shape)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
lr = LogisticRegression()
model_logic = lr.fit(x_train, y_train)
y_predict=lr.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy="minority",random_state=2020)
# fit and apply the transform
X_over, y_over = oversample.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2)
model_oversample = lr.fit(x_train, y_train)
y_predict=model_oversample.predict(x_test)
print(classification_report(y_test,y_predict))
#SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=2020).fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
model_resample = lr.fit(x_train, y_train)
y_predict=model_resample.predict(x_test)
print(classification_report(y_test,y_predict))
from imblearn.over_sampling import BorderlineSMOTE
sm = BorderlineSMOTE(random_state=2020)
X_res, y_res = sm.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
model_resample = lr.fit(x_train, y_train)
y_predict=model_resample.predict(x_test)
print(classification_report(y_test,y_predict))





