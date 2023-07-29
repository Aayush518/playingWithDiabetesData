from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import streamlit as st

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
        st.error("Invalid model selection.")
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
