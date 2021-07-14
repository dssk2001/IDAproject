import streamlit as st
import pandas as pd
import numpy as np
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
st.title("Loan Default Prediction Webapp")
st.subheader("This app is used to predict loan defaults using supervised machine learning techniques")
st.sidebar.subheader("Give the X and Y training Files")
uploaded_xfile = st.sidebar.file_uploader(label="Upload feature CSV file",type="csv")
uploaded_yfile = st.sidebar.file_uploader(label="Upload label CSV file",type="csv")
classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest","XGBoost","Ridge Classifier"))
#st.sidebar.button("Submit")
global df_x,df_y
try:
    df_x = pd.read_csv(uploaded_xfile)
    df_y = pd.read_csv(uploaded_yfile)
except Exception as e:
    pass
if(uploaded_xfile is not None and uploaded_yfile is not None):
    # Preprocess of given data
    ohe = OneHotEncoder()
    ohe_loan = ohe.fit_transform(df_x[["Loan.type"]])
    df_x = df_x.join(pd.DataFrame(ohe_loan.toarray(), columns=ohe.categories_))
    ohe_occ = ohe.fit_transform(df_x[["Occupation.type"]])
    df_x = df_x.join(pd.DataFrame(ohe_occ.toarray(), columns=ohe.categories_))
    df_x.drop(['Loan.type'],axis=1,inplace=True)
    df_x.drop(['Occupation.type'],axis=1,inplace=True)
    df_x.drop(['ID'],axis=1,inplace=True)
    df_y.drop(['ID'], axis=1, inplace=True)
    #Preprocess ends
try:
    st.write(df_x)
    st.write("Size of dataset",df_x.shape)
    st.write(df_y)
    st.write("Num of Classes",len(np.unique(df_y)))
except Exception as e:
    pass

def split(X,y):
    X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,stratify=y)
    return X_train,X_test,y_train,y_test

global X_train,X_test,y_train,y_test

try:
    X_train,X_test,y_train,y_test = split(df_x,df_y)
except Exception as e:
    pass

def train_class(classifier_namw):
    if(classifier_namw=="KNN"):
        clf = KNeighborsClassifier()
        return clf
    elif(classifier_namw=="SVM"):
        clf = SVC()
        return clf
    elif(classifier_namw=="Random Forest"):
        clf = RandomForestClassifier()
        return clf
    elif(classifier_namw=="XGBoost"):
        clf = xgboost.XGBClassifier(objective="binary:logistic",seed=42,use_label_encoder=False)
        return clf
    else:
        clf = RidgeClassifier()
        return clf

clf = train_class(classifier_name)
if st.sidebar.button("Submit"):
    try:
        clf.fit(X_train,y_train)
        acc = accuracy_score(y_test,clf.predict(X_test))
        st.write(f"classifier={classifier_name}")
        st.write(f"accuracy in testing set = {acc}")
    except Exception as e:
        st.write("XGBoost error")
        pass