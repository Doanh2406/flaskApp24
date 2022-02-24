import pandas as pd
import numpy as np
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from math import sqrt

df = pd.read_csv (r'C:\Users\quocd\Desktop\flash-backend\traindata.csv')
print (df)
df.head()
print (df.head())

df.isnull().sum()
df = df.drop(columns=['ID'])
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

df.shape
df.nunique()
df.describe()
print (df)

cat_feat = ['CONSOLE','CATEGORY', 'PUBLISHER', 'RATING']
# cat_feat = [var for var in X_train.columns if X_train[var].dtype == "O"]
features = list(set(train.columns)-set(['SalesInMillions']))
target = 'SalesInMillions'
model = cat.CatBoostRegressor(random_state=100,cat_features=cat_feat,verbose=0)

model.fit(train[features],train[target])

y_true= pd.DataFrame(data=test[target], columns=['SalesInMillions'])
test_temp = test.drop(columns=[target])

y_pred = model.predict(test_temp[features])
rmse = sqrt(mean_squared_error(y_true, y_pred))
print(rmse)

filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))