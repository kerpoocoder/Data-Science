import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

db=pd.read_excel('ToyotaCorolla.xls')
fuel=[]
for x in db['Fuel_Type']:
    if x=='CNG':
        fuel.append(0)
    elif x=='Petrol':
        fuel.append(1)
    else:
        fuel.append(2)
db['numeric_fuel']=pd.Series(fuel)

lst=list(db.columns)
lst.pop(0)
lst.pop(0)
lst.pop(0)
selected=[]
for x in lst:
    if db[x].dtype=='int64' and abs(db.corr()[x]['Price'])>0.25:
        selected.append(x)
        
X=db[selected]
Y=db['Price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size= 0.3,random_state=1)

model = LinearRegression()
model.fit(X_train, Y_train)

y_test_predicted = model.predict(X_test)
print('MSE is: ',mean_squared_error(Y_test, y_test_predicted).round(2))
print('RMSE is: ',model.score(X_test, Y_test))