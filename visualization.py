import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pdpbox
import streamlit as st
import numpy as np

def plot_feature_importance(clf, X):
    st.write("\nFeature Importance:")
    feature_importance = pd.Series(clf.feature_importances_, index=X.columns)
    feature_importance.sort_values(ascending=False, inplace=True)
    st.bar_chart(feature_importance)

def plot_partial_dependence(clf, X_train, features):
    st.write("\nPartial Dependence Plots:")
    fig, axes = pdpbox.pdp.pdp_interact_plot(model=clf, dataset=X_train, model_features=X_train.columns, features=features)
    st.pyplot(fig)

def plot_shap_values(clf, X_train):
    st.write("\nSHAP (SHapley Additive exPlanations):")
    explainer = shap.Explainer(clf)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
    st.pyplot()

def plot_feature_distribution(X):
    st.write("\nFeature Distribution:")
    for col in X.columns:
        fig, ax = plt.subplots()
        sns.histplot(X[col], kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

def plot_correlation_heatmap(X):
    st.write("\nCorrelation Heatmap:")
    corr = X.corr()
    mask = np.triu(corr)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
    st.pyplot()

def plot_line_chart():
    st.write("\nLine Chart:")
    # Add code to plot the line chart (assuming you have appropriate data for it)
    # For example, generate some random data for illustration:
    x = np.arange(1, 11)
    y = np.random.randint(10, 30, size=10)
    
    plt.plot(x, y, marker='o', color='b')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Line Chart Example")
    st.pyplot()

def plot_pie_chart(y):
    st.write("\nOutcome Distribution (Pie Chart):")
    outcome_counts = y.value_counts()
    labels = ["No Diabetes (0)", "Diabetes (1)"]
    colors = ["lightskyblue", "lightcoral"]
    plt.pie(outcome_counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.axis("equal")
    st.pyplot()

def plot_line_chart_age_bmi(X, y):
    st.write("\nLine Chart for Age and BMI:")
    fig, ax = plt.subplots()
    ax.plot(X["Age"], label="Age")
    ax.plot(X["BMI"], label="BMI")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Value")
    ax.set_title("Line Chart: Age and BMI")
    ax.legend()
    st.pyplot(fig)
