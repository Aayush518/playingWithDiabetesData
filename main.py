import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import shap
import pdpbox
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

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    if model_name == "Random Forest":
        clf = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]}
    elif model_name == "Logistic Regression":
        clf = LogisticRegression(random_state=42)
        param_grid = {"C": [0.1, 1.0, 10.0]}
    elif model_name == "SVM":
        clf = SVC(probability=True, random_state=42)
        param_grid = {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
    else:
        return

    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model:", model_name)
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Accuracy:", accuracy)

    st.write("\nClassification Report:")
    st.write(classification_report(y_test, y_pred))

    st.write("\nConfusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    plot_roc_curve(best_model, X_test, y_test)

    # Feature Importance
    if model_name == "Random Forest":
        st.write("\nFeature Importance:")
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
        feature_importance.sort_values(ascending=False, inplace=True)
        st.bar_chart(feature_importance)

    # Partial Dependence Plots (PDP) for the first two features
    st.write("\nPartial Dependence Plots:")
    if model_name == "Random Forest":
        fig, axes = pdpbox.pdp.pdp_interact_plot(model=best_model, dataset=X_train, model_features=X.columns, features=["Glucose", "BMI"])
        st.pyplot(fig)

    # SHAP (SHapley Additive exPlanations) for individual predictions
    st.write("\nSHAP (SHapley Additive exPlanations):")
    if model_name == "Random Forest":
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train)
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

    # Model Selection
    st.subheader("Model Selection:")
    model_name = st.selectbox("Select a model:", ["Random Forest", "Logistic Regression", "SVM"])
    st.write("You selected:", model_name)

    st.subheader("Model Performance:")
    train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, model_name)

if __name__ == "__main__":
    main()
