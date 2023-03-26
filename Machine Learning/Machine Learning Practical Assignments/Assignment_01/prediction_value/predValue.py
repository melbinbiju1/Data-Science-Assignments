from flask import request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


class Prediction:

    def __init__(self,data):
        self.data = data

    def predictValue(self):
        crim = self.data["crim"]
        zn = self.data["zn"]
        indus = self.data["indus"]
        chas = self.data["chas"]
        nox = self.data["nox"]
        rm = self.data["rm"]
        age = self.data["age"]
        dis = self.data["dis"]
        rad = self.data["rad"]
        tax = self.data["tax"]
        ptratio = self.data["ptratio"]
        b = self.data["b"]
        lstat = self.data["lstat"]
        

        values = [[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]]


        scaler=StandardScaler()
        x_pred = scaler.fit_transform(values)

        # Load the saved model
        model = joblib.load('linear_regression_model.pkl')

        y_pred = model.predict(x_pred)

        return y_pred
