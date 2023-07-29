import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import streamlit as st

def load_data():
    data = pd.read_csv("diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_and_evaluate(X_train, X_test, y_train, y_test, X, y):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    st.write("\nClassification Report:")
    st.write(classification_report(y_test, y_pred))

    st.write("\nConfusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    plot_roc_curve(clf, X_test, y_test)

    # Additional Features
    st.subheader("Additional Features:")

    # Pie Chart - Target Variable Distribution
    st.write("\nPie Chart - Target Variable Distribution:")
    fig, ax = plt.subplots()
    y_train.value_counts().plot.pie(autopct="%1.1f%%", labels=["Non-Diabetic", "Diabetic"], colors=['skyblue', 'lightcoral'], ax=ax)
    ax.set_title('Target Variable Distribution')
    st.pyplot(fig)

    # Line Chart - Glucose Levels vs. Age
    st.write("\nLine Chart - Glucose Levels vs. Age:")
    fig, ax = plt.subplots()
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_train_df["Age"] = X_train_df["Age"].astype(int)
    sns.lineplot(data=X_train_df, x="Age", y="Glucose", hue=y_train, palette="husl", ax=ax)
    ax.set_title('Glucose Levels vs. Age')
    st.pyplot(fig)

def plot_roc_curve(clf, X, y):
    st.write("\nROC Curve:")
    y_score = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(fig)

def main():
    st.title("Diabetes Prediction Web App")

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)

    st.write("Number of samples:", X.shape[0])
    st.write("Number of features:", X.shape[1])

    st.write("Sample Data:")
    st.write(X.head())

    st.write("Feature Distribution:")
    for col in X.columns:
        fig, ax = plt.subplots()
        sns.histplot(X[col], kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

    st.subheader("Model Performance:")
    train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, X, y)

if __name__ == "__main__":
    main()
