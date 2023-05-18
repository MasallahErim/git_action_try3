#Importing the libraries
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay




path = os.getcwd()
df = pd.read_csv(path+"/data/processedData.csv")
del df["Unnamed: 0"]

x = df.drop(["Exited"],axis=1)
y = df["Exited"]

xtrain, xtest, ytrain, ytest =  train_test_split(x,y,test_size = 0.11, random_state = 42)


modeldummy = DummyClassifier()
modeldummy.fit(xtrain, ytrain)
preddummy = modeldummy.predict(xtest)
scoredummy = modeldummy.score(xtest, ytest)
msedummy = mean_squared_error(y_pred=preddummy, y_true=ytest)


# modelLogicReg = LogisticRegression()
# modelLogicReg.fit(xtrain, ytrain)
# predlogic = modelLogicReg.predict(xtest)
# scorelogic = modelLogicReg.score(xtest, ytest)
# mselogic = mean_squared_error(y_pred=predlogic, y_true=ytest)



dummyProbs = modeldummy.predict_proba(xtest)[:, 1]

# Plot ROC
fprDummy, tprDummy, thresholdsDummy = metrics.roc_curve(ytest, dummyProbs)
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(fprDummy, tprDummy, label = "Dummy")
axes.set_xlabel("False positive rate")
axes.set_ylabel("True positive rate")
axes.set_title("Dummy")
axes.grid(which = 'major', c='#cccccc', linestyle='--', alpha=0.5)
axes.legend(shadow=True)
plt.savefig('ROC.png', dpi=120)






# - - - - - - - GENERATE METRICS FILE
with open("metrics.txt", 'w') as outfile:
        json.dump(
        	{ "score": scoredummy, "mse": msedummy,}, 
        	  outfile
        	)





























































