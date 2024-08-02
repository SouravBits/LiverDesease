import pandas as pd
import numpy as np
import joblib
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


# Please replace the location with actual filepath
location = "liver_disease_1.csv" 
data = pd.read_csv(location)
data.shape

# Step 1 : print 2 rows for sanity check
print(data.head(2))


####

# Data Insights

# Count the number of samples in each class
class_counts = data['Dataset'].value_counts()
print(class_counts)


#column list
data.columns


####
# Encoding the target variable from yes/no to 1/0

# Sample DataFrame
df = pd.DataFrame(data)

# Import LabelEncoder from scikit-learn
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the 'Yes_No' column
df['Dataset_Encoded'] = label_encoder.fit_transform(df['Dataset'])

# Display the DataFrame with label encoding
df=df.drop(['Dataset'],axis=1)
df.shape


# based on above co relation analysis


df=df.drop(['Total_Bilirubin'],axis=1)
df=df.drop(['Alamine_Aminotransferase'],axis=1)
df=df.drop(['Total_Protiens'],axis=1)
df=df.drop(['Albumin_and_Globulin_Ratio'],axis=1)
df.columns

# print 2 rows after column drop
print('---------- After droppping columns ----------')
print(df.head(2))

#Normalization with minmax scaler

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# Filling the null values based on the KNN imputer so that the nulls get replaced by the appropriate values.

# Separate the data into two DataFrames based on the target variable
df_class_0 = df.loc[df['Dataset_Encoded'] == 0]
df_class_1 = df.loc[df['Dataset_Encoded'] == 1]

print('df_class_0.shape ',df_class_0.shape)
print('df_class_1.shape ',df_class_1.shape)

# Impute missing values separately for each class using KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)

# Impute for class 0
imputer = KNNImputer(n_neighbors=5)
df_class_0 = pd.DataFrame(imputer.fit_transform(df_class_0),columns = df.columns)
df_class_1 = pd.DataFrame(imputer.fit_transform(df_class_1),columns = df.columns)
df_imputed = pd.concat([df_class_0, df_class_1])

# Split the data into features (X) and target (y)

# Extract y vector from the dataframe
Y = df_imputed['Dataset_Encoded']
X = df_imputed.drop(['Dataset_Encoded'], axis=1)


# spliting train and test dataset (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# spliting train and test dataset (85:15)
X_trainb, X_testb, Y_trainb, Y_testb = train_test_split(X, Y, test_size=0.15, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Tune hyperparameters using GridSearchCV 
# experimenting with number of trees 5,10,20,30 and maximum depth 3,5,7. 
# Since we have just 6 features after data cleaning and limited samples the above chosen trees and depth seems ok.
 
param_grid = {
    'n_estimators': [5, 10, 20, 30],
    'max_depth': [3, 5, 7],
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters found from above grid search.
print('Best hyperparameters:',  grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

# Evaluate the Random Forest model
#y_pred_rf = best_rf_model.predict(X_test)

# Print evaluation metrics
#print("Random Forest Model [80-20 split results]:")
#print("Accuracy:", accuracy_score(Y_test, y_pred_rf))
#print('\n')
#print(confusion_matrix(Y_test, y_pred_rf))
#print('\n')
#print("Classification Report:\n", classification_report(Y_test, y_pred_rf))

print('Training Liver disease model completed')
print('saving model')

joblib.dump(best_rf_model,'best_rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print('Model saved')

