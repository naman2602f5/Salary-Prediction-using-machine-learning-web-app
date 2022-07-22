import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask,request,render_template
import os

PEOPLE_FOLDER = os.path.join('static', 'pic')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/predict',methods=['GET','POST'])
def display():
    data_path = 'E:\\PythonProjects\\Salary Prediction\\'
    data=pd.read_csv(data_path + 'Salary.csv')
    print(data)

    print(data.describe())

    X = data.iloc[:,:-1].values
    y=data.iloc[:,-1].values

    print(X.shape,y.shape)

    X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=101,test_size=0.2)
    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

    lr=LinearRegression()
    lr.fit(X_train,Y_train)

    pred=lr.predict(X_test)
    print(pred)

    print(Y_test)

    diff=Y_test - pred
    pd.DataFrame(np.c_[Y_test,pred,diff],columns=['Actual','Predicted','Difference'])

    print(lr.score(X_test,Y_test))

    rmse=np.sqrt(mean_squared_error(Y_test,pred))
    print(rmse)

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg')
    if(request.method=='POST'):
        exp = request.form.get('YearsExperience', type=int)
        val=lr.predict([[exp]])[0]
        return render_template('index.html',user_image = full_filename, prediction_text=f"Salary of employee having {exp} experiences is {int(val)} thousands") 
    else:
        return render_template('index.html',user_image = full_filename)

if __name__ == "__main__":
    app.run(debug=True)


