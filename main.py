import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df=pd.read_csv("2 loan_approval_data.csv")

#Dealing MISSING DATA
numerical_col=df.select_dtypes(include=["number"]).columns
categorical_col=df.select_dtypes(include=["object"]).columns
from sklearn.impute import SimpleImputer
num_imp=SimpleImputer(strategy="mean")
df[numerical_col]=num_imp.fit_transform(df[numerical_col])
cat_imp=SimpleImputer(strategy="most_frequent")
df[categorical_col]=cat_imp.fit_transform(df[categorical_col])

#EDA
"""
classes_count=df["Loan_Approved"].value_counts()
plt.pie(classes_count,labels=["No","Yes"],autopct="%1.1f%%")
plt.title("Loan Approved")
#plt.show()

edu_cnt=df["Education_Level"].value_counts()
ax=sns.barplot(edu_cnt)
ax.bar_label(ax.containers[0])

sns.histplot(
    data=df,
    x="Applicant_Income",
    bins=20
)

#we use boxplot for getting outliers
fig,axes=plt.subplots(2,2)
sns.boxenplot(ax=axes[0,0],data=df,x="Loan_Approved",y="Applicant_Income")
sns.boxenplot(ax=axes[0,1],data=df,x="Loan_Approved",y="Credit_Score")
sns.boxenplot(ax=axes[1,0],data=df,x="Loan_Approved",y="DTI_Ratio")
sns.boxenplot(ax=axes[1,1],data=df,x="Loan_Approved",y="Savings")
plt.show()
"""

#Feature ENCODING
df=df.drop(columns=["Applicant_ID"],axis=1)

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
#LABEL ENCODING
le=LabelEncoder()
df["Education_Level"]=le.fit_transform(df["Education_Level"])
df["Loan_Approved"]=le.fit_transform(df["Loan_Approved"])

#ONEHOT ENCODING
cols=["Employment_Status","Property_Area","Marital_Status","Loan_Purpose","Gender","Employer_Category"]
ohe=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")
encoded=ohe.fit_transform(df[cols])
encoded_df=pd.DataFrame(encoded,columns=ohe.get_feature_names_out(cols),index=df.index)
df=pd.concat([df,encoded_df],axis=1)
df=df.drop(columns=cols)
"""
#CORRELATION HEATMAP
num_col=df.select_dtypes(include="number")
corr_matrix=num_col.corr()
#print(num_col.corr()["Loan_Approved"].sort_values(ascending=False))

sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm")

plt.show()
"""

#FEATURE ENGG
df["DTI_Ratio_sq"]=df["DTI_Ratio"]**2
df["Credit_Score_sq"]=df["Credit_Score"]**2
#for dealing with extream/skewed data we take x-->log(1-x)
df["Applicant_Income_log"]=np.log1p(df["Applicant_Income"])

#TRAIN TEST SPLIT
x=df.drop(columns=["Loan_Approved","DTI_Ratio","Credit_Score","Applicant_Income"])
y=df["Loan_Approved"]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#TRAINING & MODEL EVALUTION 
#-----Logistic Regression--------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score, confusion_matrix
log_model = LogisticRegression()
log_model.fit(x_train_scaled,y_train)
y_pred_log=log_model.predict(x_test_scaled)
#use presicion to eliminate false positive
#use Recall for false negative
#F1 scole is balance of F1,Precesion
print("Logistic Regression")
print("Precesion : ", precision_score(y_test,y_pred_log))
print("CM : ",confusion_matrix(y_test,y_pred_log))


#KNN MODEL
from sklearn.neighbors import KNeighborsClassifier
Knn_model = KNeighborsClassifier(n_neighbors=3)
Knn_model.fit(x_train_scaled,y_train)
y_pred_knn=Knn_model.predict(x_test_scaled)
print("KNN Classifier")
print("Precesion : ", precision_score(y_test,y_pred_knn))
print("CM : ",confusion_matrix(y_test,y_pred_knn))

#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
NB_model= GaussianNB()
NB_model.fit(x_train_scaled,y_train)
y_pred_NB=NB_model.predict(x_test_scaled)
print("NB model")
print("Precesion : ", precision_score(y_test,y_pred_NB))
print("CM : ",confusion_matrix(y_test,y_pred_NB))

dct={"Logistic Regression":precision_score(y_test,y_pred_log),"KNN Classifier":precision_score(y_test,y_pred_knn),"Naive Bayes Algorithm":precision_score(y_test,y_pred_NB)}


for key,value in dct.items():
    print(key," have precision of : ",value*100,"%")