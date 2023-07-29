import streamlit as st
from data import load_data, split_data
from preprocess import feature_scaling
from models import train_and_evaluate
import visualization

def main():
    st.title("Diabetes Prediction Web App")

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)

    st.write("Number of samples:", X.shape[0])
    st.write("Number of features:", X.shape[1])

    st.write("Sample Data:")
    st.write(X.head())

    visualization.plot_feature_distribution(X)

    # Model Selection
    st.subheader("Model Selection:")
    model_name = st.selectbox("Select a model:", ["Random Forest", "Logistic Regression", "SVM"])
    st.write("You selected:", model_name)

    st.subheader("Model Performance:")
    train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, model_name)

    # Visualizations
    st.subheader("Visualizations:")
    if model_name == "Random Forest":
        visualization.plot_feature_importance(best_model, X)
    elif model_name == "Logistic Regression":
        features = st.multiselect("Select features for Partial Dependence Plot:", X.columns)
        if features:
            visualization.plot_partial_dependence(best_model, X_train_scaled, features)
    elif model_name == "SVM":
        visualization.plot_shap_values(best_model, X_train_scaled)

    visualization.plot_correlation_heatmap(X)

    # You can add more visualizations as needed (e.g., line chart) using the functions in visualization.py

if __name__ == "__main__":
    main()
