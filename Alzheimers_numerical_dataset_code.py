#!/usr/bin/env python
# coding: utf-8

# # Imporing libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Reading csv and Exploring the dataset

# In[2]:


d1=pd.read_csv('/Users/srujithadevineni/research/urop/alzehemier_disease/Dataset_numerical/oasis_longitudinal.csv')
d1.head()


# In[3]:


d2=pd.read_csv('/Users/srujithadevineni/research/urop/alzehemier_disease/Dataset_numerical/oasis_cross-sectional.csv')
d2.head()


# In[4]:


d2.ID.isnull().sum()


# In[5]:


d1[['MRI ID']].isnull().sum()


# In[6]:


d2.shape


# In[7]:


d1.shape


# In[8]:


d2.ID.nunique()


# In[9]:


d1[['MRI ID']].nunique()


# In[10]:


d1


# In[11]:


d1.Group.unique()


# # Check missing values by each column

# In[12]:


pd.isnull(d1).sum() 


# # Filled null value with column median

# In[13]:


d1["SES"].fillna(d1["SES"].median(), inplace=True)
d1["MMSE"].fillna(d1["MMSE"].median(), inplace=True)


# # Bar-graph showing how many people have Alzheimer
# ## *same person visits two or more time so we take the single visit data

# In[14]:


sns.set_style("whitegrid")
ex_d1 = d1.loc[d1['Visit'] == 1]
palette=sns.color_palette("terrain")
sns.countplot(x='Group', data=ex_d1,palette=palette)
print(palette[2])


# In[15]:


ex_d1['Group'] = ex_d1['Group'].replace(['Converted'], ['Demented'])
d1['Group'] = d1['Group'].replace(['Converted'], ['Demented'])
sns.countplot(x='Group', data=ex_d1,palette=palette)


# # Bar-graph showing Male vs female demented rate  

# In[16]:


# bar drawing function
def bar_chart(feature):
    Demented = ex_d1[ex_d1['Group']=='Demented'][feature].value_counts()
    Nondemented = ex_d1[ex_d1['Group']=='Nondemented'][feature].value_counts()
    df_bar = pd.DataFrame([Demented,Nondemented])
    df_bar.index = ['Demented','Nondemented']
    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))
    print(df_bar)
                
                
# Gender  and  Group ( Female=0, Male=1)
bar_chart('M/F')

plt.xlabel('Group',fontsize=13)
plt.xticks(rotation=0,fontsize=12)
plt.ylabel('Number of patients',fontsize=13)
plt.legend()
plt.title('Gender and Demented rate',fontsize=14)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[18]:


#convet the charecter data into numeric
group_map = {"Demented": 1, "Nondemented": 0}

d1['Group'] = d1['Group'].map(group_map)
d1['M/F'] = d1['M/F'].replace(['F','M'], [0,1])


# # spliting data into train(80percent) and test(20percent)

# In[19]:


from sklearn.model_selection import train_test_split

feature_col_names = ["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
predicted_class_names = ['Group']

X = d1[feature_col_names].values
y = d1[predicted_class_names].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# # Confusion Metrics Table

# In[81]:


from sklearn import metrics
def plot_confusion_metrix(y_test,model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['Nondemented','Demented']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), color='black')
    plt.show()


# # Roc curve

# In[82]:


from sklearn.metrics import roc_curve, auc
def report_performance(model):

    model_test = model.predict(X_test)

    print("\n\nConfusion Matrix:")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("\n\nClassification Report: ")
    print(metrics.classification_report(y_test, model_test))
    cm = metrics.confusion_matrix(y_test, model_test)
    plot_confusion_metrix(y_test, model_test)

total_fpr = {}
total_tpr = {}
def roc_curves(model):
    predictions_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(predictions_test,y_test)
    roc_auc = auc(fpr, tpr)
    total_fpr[str((str(model).split('(')[0]))] = fpr
    total_tpr[str((str(model).split('(')[0]))] = tpr
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# # Accuracy code

# In[83]:


total_accuracy = {}
def accuracy(model):
    pred = model.predict(X_test)
    accu = metrics.accuracy_score(y_test,pred)
    print("\nAcuuracy Of the Model: ",accu,"\n\n")
    total_accuracy[str((str(model).split('(')[0]))] = accu


# In[84]:


# Replace NaN values with an empty string
d1.fillna("", inplace=True)


# ### RandomForestClassifier

# In[85]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini', max_depth=8, max_features=0.5, n_estimators=200)


param_grid = { 
    'n_estimators': [200],
    'max_features': ['auto'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini']
}

#CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5,scoring = 'roc_auc')
rfc.fit(X_train, y_train.ravel())
#print("Best parameters set found on development set:")
#print(rfc.best_params_)
report_performance(rfc) 
roc_curves(rfc)
accuracy(rfc)

#feat_importances = pd.Series(rfc.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### SVM

# In[86]:


from sklearn.svm import SVC
svm = SVC(kernel="linear", C=0.1,random_state=0)
svm.fit(X_train, y_train.ravel())
report_performance(svm) 
roc_curves(svm)
accuracy(svm)
#feat_importances = pd.Series(svm.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### DecisionTreeClassifier

# In[87]:


from sklearn.tree import DecisionTreeClassifier
clf_dtc = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
clf_dtc.fit(X_train, y_train.ravel())
report_performance(clf_dtc) 
roc_curves(clf_dtc)
accuracy(clf_dtc)
#importances = clf.feature_importances_


#feat_importances = pd.Series(clf_dtc.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### XGB Classifier 

# In[88]:


from xgboost import XGBClassifier
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [1,2,3,4,5]
        }

clf_xgb = XGBClassifier(random_state=0)
clf_xgb.fit(X_train, y_train.ravel())
report_performance(clf_xgb) 
roc_curves(clf_xgb)
accuracy(clf_xgb)

#feat_importances = pd.Series(clf_xgb.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### Gaussian naive_bayes classifier

# In[89]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train.ravel())
report_performance(gnb) 
roc_curves(gnb)
accuracy(gnb)
#importances = clf.feature_importances_


#feat_importances = pd.Series(gnb.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### Bernoulli naive_bayes classifier

# In[90]:


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, y_train.ravel())
report_performance(clf) 
roc_curves(clf)
accuracy(clf)
#importances = clf.feature_importances_


#feat_importances = pd.Series(clf.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### AdaBoostClassifier

# In[91]:


from sklearn.ensemble import AdaBoostClassifier
adb= AdaBoostClassifier(n_estimators=100, random_state=0)
adb.fit(X_train, y_train.ravel())
report_performance(adb) 
roc_curves(adb)
accuracy(adb)
#importances = clf.feature_importances_


#feat_importances = pd.Series(adb.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### LogisticRegression

# In[92]:


from sklearn.linear_model import LogisticRegression

logRegModel = LogisticRegression()
logRegModel.fit(X_train, y_train.ravel())
report_performance(logRegModel) 
roc_curves(logRegModel)
accuracy(logRegModel)
#importances = clf.feature_importances_


#feat_importances = pd.Series(logRegModel.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# ### knn Classifier

# In[93]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train.ravel())
report_performance(neigh) 
roc_curves(neigh)
accuracy(neigh)
#importances = clf.feature_importances_

#feat_importances = pd.Series(neigh.feature_importances_, index=feature_col_names)
#feat_importances.nlargest(8).plot(kind='barh')
#plt.title("Feature Importance:")
#plt.show()


# # Comparison of different Model vs Accuracy

# In[95]:


data = total_accuracy.values()
labels = total_accuracy.keys()


plt.plot([i for i, e in enumerate(data)], data, 'ro'); plt.xticks([i for i, e in enumerate(labels)], [l[0:25] for l in labels])
plt.title("Model Vs Accuracy",fontsize = 12)
plt.xlabel('Model',fontsize = 12)
plt.xticks(rotation = 50)
plt.ylabel('Accuracy',fontsize = 12)

