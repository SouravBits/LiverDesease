import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN

# Replace with your actual file path
location = "liver_disease_1.csv" 
data = pd.read_csv(location)

# Step 1 : print 2 rows for sanity check
print(data.head(2))

# Data Insights
class_counts = data['Dataset'].value_counts()
print(class_counts)

# Encoding the target variable from yes/no to 1/0
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['Dataset_Encoded'] = label_encoder.fit_transform(df['Dataset'])
df = df.drop(['Dataset'], axis=1)

# Dropping less significant columns based on correlation analysis
df = df.drop(['Total_Bilirubin', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin_and_Globulin_Ratio'], axis=1)
print('---------- After dropping columns ----------')
print(df.head(2))

# Normalization with MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Imputing missing values using KNNImputer
df_class_0 = df.loc[df['Dataset_Encoded'] == 0]
df_class_1 = df.loc[df['Dataset_Encoded'] == 1]

imputer = KNNImputer(n_neighbors=5)
df_class_0 = pd.DataFrame(imputer.fit_transform(df_class_0), columns=df.columns)
df_class_1 = pd.DataFrame(imputer.fit_transform(df_class_1), columns=df.columns)
df_imputed = pd.concat([df_class_0, df_class_1])

# Split the data into features (X) and target (y)
Y = df_imputed['Dataset_Encoded']
X = df_imputed.drop(['Dataset_Encoded'], axis=1)

# Splitting train and test dataset (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLflow tracking
mlflow.set_experiment("Liver Disease Detection")
param_grid = {
    'n_estimators': [5, 10, 20, 30],
    'max_depth': [3, 5, 7],
}

for run in range(3):  # Run the experiment 3 times
    with mlflow.start_run(run_name=f"run_{run+1}"):
        rf_classifier = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, Y_train)
        
        best_rf_model = grid_search.best_estimator_
        y_pred_rf = best_rf_model.predict(X_test)
        
        accuracy = accuracy_score(Y_test, y_pred_rf)
        clf_report = classification_report(Y_test, y_pred_rf, output_dict=True)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(best_rf_model, "model")
        
        # Log classification report metrics
        for label, metrics in clf_report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", metric_value)
        
        print(f"Run {run+1} - Best hyperparameters: {grid_search.best_params_}")
        print(f"Run {run+1} - Accuracy: {accuracy}")

        # Save model and scaler
        joblib.dump(best_rf_model, f'best_rf_model_run_{run+1}.joblib')
        joblib.dump(scaler, f'scaler_run_{run+1}.joblib')
        print(f"Run {run+1} - Model saved")

print('Training Liver disease model completed')
