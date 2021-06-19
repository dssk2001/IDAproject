# Importing the required libraries
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams as rp

# Preprocessing of the given data begins
ohe = OneHotEncoder()
df_x = pd.read_csv('train_x_imputed.csv')

ohe_loan = ohe.fit_transform(df_x[["Loan.type"]])
df_x = df_x.join(pd.DataFrame(ohe_loan.toarray(), columns=ohe.categories_))

ohe_occ = ohe.fit_transform(df_x[["Occupation.type"]])
df_x = df_x.join(pd.DataFrame(ohe_occ.toarray(), columns=ohe.categories_))

df_x.drop(['Loan.type'],axis=1,inplace=True)
df_x.drop(['Occupation.type'],axis=1,inplace=True)
df_x.drop(['ID'],axis=1,inplace=True)
df_x.to_csv('new.csv',index=False)
# Preprocessing of Training csv file ends

# Preprocessing of test csv file begins
df_x_test = pd.read_csv('test_x.csv')

ohe_loan = ohe.fit_transform(df_x_test[["Loan type"]])
df_x_test = df_x_test.join(pd.DataFrame(ohe_loan.toarray(), columns=ohe.categories_))

ohe_occ = ohe.fit_transform(df_x_test[["Occupation type"]])
df_x_test = df_x_test.join(pd.DataFrame(ohe_occ.toarray(), columns=ohe.categories_))

df_x_test.drop(['Loan type'],axis=1,inplace=True)
df_x_test.drop(['Occupation type'],axis=1,inplace=True)
df_x_test.drop(['ID_Test'],axis=1,inplace=True)
df_x_test.to_csv('X_test.csv',index=False)
# Preprocessing of data ends

df_x_test = pd.read_csv('test_x.csv')
arr = df_x_test.loc[:,'ID_Test'].values

# Conversion of dataframes to numpy arrays
df_x_tr = pd.read_csv('new.csv')

df_y = pd.read_csv('train_y_imputed.csv')

df_y.drop(['ID'],axis=1,inplace=True)

tuples_x = [tuple(x) for x in df_x_tr.values]
tuples_y = [tuple(y) for y in df_y.values]
tuples_y = np.ravel(tuples_y)

X = np.asarray(tuples_x)
y = np.asarray(tuples_y)


x_train,x_val,y_train,y_val = tts(X,y,test_size=0.1,random_state=42) # Splitting the data with test ratio of 0.1

'''
This info has been taken from the documentation of XGBoost package.
'''
# Creating a DMatrix for training the Extreme Gradient Classifier
dtrain = xgboost.DMatrix(x_train,label=y_train,feature_names=['Expense','Income','Age','Score1','Score2','Score3','Score4','Score5','A','B','X','Y','Z'])
dval = xgboost.DMatrix(x_val,label=y_val,feature_names=['Expense','Income','Age','Score1','Score2','Score3','Score4','Score5','A','B','X','Y','Z'])
evallist = [(dtrain, 'train'), (dval, 'eval')]

# Creating the model instance
xgb = xgboost.train({'max_depth':25,'objective':'binary:logistic','gamma':10},dtrain,1000,evals=evallist,early_stopping_rounds=20)

# Predicting on the training split of data
y_pred_train = xgb.predict(dtrain) # This returns the probability P(Y=k|X) where k is class
y_predictions_train = (y_pred_train > 0.5).astype(int)  # Using a threshold of 0.5 we predict the class

# Predicting on the test split of data
y_pred = xgb.predict(dval,ntree_limit=xgb.best_ntree_limit)
y_predictions = (y_pred>0.5).astype(int)

print("Accuracy in training set is :",accuracy_score(y_predictions_train,y_train))
print("Accuracy in test set is :",accuracy_score(y_predictions,y_val))


# Predicting on the test csv file and writing predictions to csv file

df_x_test = pd.read_csv('X_test.csv')
tuples_x = [tuple(x) for x in df_x_test.values]
X_test = np.asarray(tuples_x)
Dtest = xgboost.DMatrix(X_test,feature_names=['Expense','Income','Age','Score1','Score2','Score3','Score4','Score5','A','B','X','Y','Z'])
Y_test_pred = xgb.predict(Dtest)
Y_test_predictions = (Y_test_pred>0.5).astype(int)
a = np.asarray(Y_test_predictions)

# Writing the ID_Test column and predictions to the CSV file
df_pred = pd.DataFrame()
df_pred.insert(0,"ID_Test",arr)
df_pred.insert(1,"Y_Predicted",a)
header = ["ID_Test","Y_Predicted"]
df_pred.to_csv('Predictions.csv',sep=',',columns=header,index=False)

# Plotting importance
xgboost.plot_importance(xgb)

# Plotting tree

xgboost.plot_tree(xgb,num_trees=2)
xgboost.to_graphviz(xgb, num_trees=2)

fig = plt.gcf()
fig.set_size_inches(150, 150)
fig.savefig('tree.png')


