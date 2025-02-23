import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


def load_data(filepath):
    df = pd.read_csv('diabetes.csv') # Load the dataset
    print(df.head(10)) # Display the first few rows
    print(df.isnull().sum()) # Check for missing values
    print(df.describe()) # Basic statistics
    return df

def preprocess_data(df):
    # Replace zeros with NaN for columns that cannot be zero
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_columns] = df[zero_columns].replace(0, np.nan)

    # Fill missing values with the median
    df.fillna(df.median(), inplace=True)

    # Check for missing values again
    print(df.isnull().sum())
    return df

def split_and_scale_data(df):
    # Separate features (X) and target (y)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    #Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"{model_name} Performance")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred ))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# Main script
def main():
    df= load_data("diabetes.csv")
    df= preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_scale_data(df)

    # EDA 
    sns.countplot(x="Outcome", data=df)
    plt.title("Distribution of Diabetes Outcomes")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    sns.pairplot(df, hue="Outcome", vars=["Glucose", "BMI", "Age"])
    plt.show()

    #train and evaluate models
    log_reg = LogisticRegression(random_state=42)
    train_and_evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")
   
    svm = SVC(probability=True, random_state=42) # Enable probability for ROC curve
    train_and_evaluate_model(svm, X_train, X_test, y_train, y_test, "Support Vector Machine")

if __name__ == "__main__":
    main()