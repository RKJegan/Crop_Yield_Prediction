# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# importing dataset                                   
dataset = pd.read_csv("TN yield data.csv")
print(dataset.shape)
print(dataset.head(5))

#Remove missing values                               
dataset = dataset.dropna(subset=["Production"])
print(dataset.shape)

#Convert prodection to classes
dataset["Production_level"] = pd.qcut(dataset["Production"],q=3,labels=["Low","Medium","High"])

#Encode categorical value                            
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

dataset["State_Name"] = le1.fit_transform(dataset["State_Name"])
dataset["District_Name"] = le2.fit_transform(dataset["District_Name"])
dataset["Season"] = le3.fit_transform(dataset["Season"])
dataset["Crop"] = le4.fit_transform(dataset["Crop"])


#split X & Y
x = dataset[["State_Name", "District_Name" ,"Crop_Year" ,"Season", "Crop", "Area"]]
y = dataset["Production_level"]

# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
print(x_train.shape)
print(y_train.shape)

#create model
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(x_train,y_train)

#predict
y_pred = model.predict(x_test)


#Accuracy_score,confusion_matrix,classification_report
print("Accuracy              : ",accuracy_score(y_test,y_pred))
print("Confusion Matrix      : ",confusion_matrix(y_test,y_pred))
print("Classification Report : ",classification_report(y_test,y_pred))

#Visualization Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=x.columns, class_names=["Low","Medium","High"], filled = True)
plt.title("Crop Production Tree")
plt.show()


#Getting user input
print("Datas are 'CASE SENSITIVE'")
state = "Tamil Nadu"
dist =       input("Enter the District :")
year =   int(input("Enter crop year    :"))
season =     input("Enter Season       :")
crop =       input("Enter crop name    :")
area = float(input("Enter Area         :"))

#convert input to encoded value
new_data = pd.DataFrame([{
    "State_Name" : le1.transform([state])[0],
    "District_Name" : le2.transform([dist])[0],
    "Crop_Year" : year,
    "Season" : le3.transform([season])[0],
    "Crop" : le4.transform([crop])[0],
    "Area" : area
}])

#prediction
prediction = model.predict(new_data)

print("\nPredicted Production Level :",prediction[0])