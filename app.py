import base64
import os
import pickle
import pandas as pd
import seaborn as sns
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind 
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import yaml
import requests
from io import BytesIO


st.write(f"Streamlit version: {st.__version__}")
st.write(f"scikit-learn version: {sklearn.__version__}")
st.write(f"joblib version: {joblib.__version__}")

# GitHub raw file URL (make sure to replace with your own URL)
github_url = "https://github.com/benlim2002/Breast-Cancer-Survivabilit-RFC/blob/main/%20rfc_model_rf_rfe.pkl"

# Fetch the .pkl file from GitHub
response = requests.get(github_url)

if response.status_code == 200:
    # Load the model from the response content
    model_data = BytesIO(response.content)
    rfc_model_rf_rfe = joblib.load(model_data)
    st.success("Model loaded successfully!")
else:
    st.error("Failed to download the model file from GitHub.")

#list of features
selected_features = ['age_at_diagnosis', 'nottingham_prognostic_index',
                     'overall_survival_months', 'jak1', 'notch3', 'map4', 'lama2', 'ar',
                     'cdkn2c', 'hsd17b11']

#transpose data
def preprocess_and_filter(data):
    data_filtered = data[selected_features]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_filtered) 
    return data_scaled

#prediction
def make_prediction(model, data):
    processed_data = preprocess_and_filter(data)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None
    return prediction, probability

#to predict survive or not chart
def plot_probability_distribution(probability, prediction):
    if prediction[0] == 0:
        status = "Not Survive"
        color = 'green'
    else:
        status = "Survive"
        color = 'red'
    
    labels = ['Not Survive', 'Survive']
    values = [probability[0][0], probability[0][1]]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=[color, 'green'])
    ax1.axis('equal') 
    ax1.set_title('Prediction Probability Distribution')
    return fig1


#second page functions
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Survival', 'Non-Survival'], 
                yticklabels=['Survival', 'Non-Survival'], cbar=False, annot_kws={"size": 16}, ax=ax)
    
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig



with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


model_config = config.get("models", {})
model_directory = config.get("paths", {}).get("model_directory", "./models/")

os.makedirs(model_directory, exist_ok=True)


custom_model = 'METABRIC_pre-processed.csv'
data_model_build = pd.read_csv(custom_model)


def preprocess_data_for_survival(data_model_build):
    X = data_model_build.drop(['overall_survival'], axis=1)
    y = data_model_build['overall_survival']
    return X, y

#t-test
def perform_t_test(X, y, p_value_threshold):
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X_numeric = X[numeric_cols]

    group1 = X_numeric[y == 0]
    group2 = X_numeric[y == 1]

    t_test_results = {}
    for col in X_numeric.columns:
        t_stat, p_value = ttest_ind(group1[col], group2[col], nan_policy='omit')
        t_test_results[col] = p_value

    significant_features = [col for col, p in t_test_results.items() if p < p_value_threshold]

    return significant_features


def perform_rfe_with_estimator(X_train, y_train, estimator="Random Forest", num_features=10):
    
    if estimator == "Random Forest":
        estimator_model = RandomForestClassifier(random_state=42)
    elif estimator == "SVM":
        estimator_model = SVC(kernel="linear")

    rfe = RFE(estimator=estimator_model, n_features_to_select=num_features)
    rfe.fit(X_train, y_train)

    selected_features = X_train.columns[rfe.support_]
    return selected_features


def initialize_model(model_name):
    model_dict = {
        "SVC": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    return model_dict.get(model_name)

#function to train SVC model
def train_svc(X_train, y_train, X_test, y_test, config):
    model_config = config["models"]["model_1"]  #update key to access SVC model
    model_name = model_config["name"]
    param_grid = model_config["param_grid"]

    #initialize and train SVC using GridSearchCV
    model = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    #get the best model
    best_model = grid_search.best_estimator_

    #evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{model_name} trained")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy on test set: {accuracy:.4f}")
    
    return best_model, accuracy


#function to train Random Forest model
def train_rfc(X_train, y_train, X_test, y_test, config):
    model_config = config["models"]["model_2"] 
    model_name = model_config["name"]
    param_grid = model_config["param_grid"]

    #initialize and train RFC using GridSearchCV
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    #get the best model
    best_model = grid_search.best_estimator_

    #evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{model_name} trained")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy on test set: {accuracy:.4f}")
    
    return best_model, accuracy


#third page functions
def preprocess_and_filter_all_features(data, model):
    # Determine which features the model expects
    expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    
    if expected_features is None:
        st.error("The model does not have feature names stored. Please ensure your model includes 'feature_names_in_'.")
        return None, None

    # Filter data to only include expected features
    data_filtered = data[expected_features]

    numeric_features = [feature for feature in expected_features if data_filtered[feature].dtype in ['int64', 'float64']]
    categorical_features = [feature for feature in expected_features if data_filtered[feature].dtype == 'object']

    # Handle numeric features
    if numeric_features:
        numeric_data = data_filtered[numeric_features]
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_data)
        
    # Handle categorical features
    if categorical_features:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_encoded = encoder.fit_transform(data_filtered[categorical_features])
        categorical_feature_names = encoder.get_feature_names_out(categorical_features)
    else:
        categorical_encoded = np.array([]).reshape(data_filtered.shape[0], 0)
        categorical_feature_names = []

    # Combine numeric and categorical
    processed_data = np.hstack([numeric_scaled, categorical_encoded]) if numeric_features and categorical_features else \
                     (numeric_scaled if numeric_features else categorical_encoded)
    feature_names = numeric_features + list(categorical_feature_names)

    return processed_data, feature_names

def make_prediction_all_features(model, data):
    processed_data, feature_names = preprocess_and_filter_all_features(data, model)
    if processed_data is None:
        return None, None, None
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None
    return prediction, probability, feature_names

def plot_probability_distribution_custom(probability, prediction):
    if prediction[0] == 0:
        status = "Not Survive"
        color = 'green'
    else:
        status = "Survive"
        color = 'red'
    
    labels = ['Not Survive', 'Survive']
    values = [probability[0][0], probability[0][1]]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=[color, 'green'])
    ax1.axis('equal') 
    ax1.set_title('Prediction Probability Distribution')
    return fig1


#streamlit app
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("### Select a Page:")

page = st.sidebar.radio(
    "Go to", 
    ["Prediction", "Custom Model Creation", "Test Your Model"],
    index=0, 
    help="Choose between making predictions or creating a custom model"
)


if page == "Prediction":
    
    st.title('Breast Cancer Survival Prediction')
    st.markdown('*****Upload a CSV file to make predictions.*****')

    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            # Check if the data is empty
            if data.empty:
                st.error("Error: The uploaded file is empty. Please upload a valid CSV file with data.")
            
            else:
                if data.shape[0] > data.shape[1]:
                    data = data.T 
                    data.columns = data.iloc[0]  
                    data = data[1:]  
                    st.write("Data was transposed. Now it's back to the original form.")
                    
                    data.reset_index(inplace=True)
                    data.rename(columns={'index': 'age_at_diagnosis'}, inplace=True)  

                st.write("Original Data:", data.head()) 
                
                
                missing_features = [feature for feature in selected_features if feature not in data.columns]
                if missing_features:
                    st.error(f"Error: The CSV file is missing the following required columns: {', '.join(missing_features)}. \n PLEASE UPLOAD ONLY CLINICAL DATA")
                else:
                    features_in_data = [feature for feature in selected_features if feature in data.columns]
                    st.subheader("Features available for prediction:")
                    for feature in features_in_data:
                        st.write(f"{feature} ... ✔️")
                    
                    #short explanation
                    st.subheader("Explanation of Key Features:") 
                    st.write(""" 
                        - **Age at Diagnosis**: Age can be an important factor in determining cancer prognosis.
                        - **Nottingham Prognostic Index**: A widely used index for breast cancer prognosis.
                        - **Jak1, Notch3, Map4, Lama2, AR, CDKN2C, HSD17B11**: Gene expression markers, with variations influencing survival rates.
                    """)
                    
                    if st.button('Predict'):
                        with st.spinner('Making prediction...'):
                            try:
                                prediction, probability = make_prediction(rfc_model_rf_rfe, data)

                                st.subheader('Prediction Results')

                                #display prediction results
                                if prediction[0] == 0:
                                    st.markdown(
                                        "<h1 style='color:red;'>This patient is predicted to NOT survive.</h3>", 
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        f"<p style='font-size:26px;'>🛑 <b>Probability of NOT SURVIVING:</b> {probability[0][0] * 100:.2f}%</p>", 
                                        unsafe_allow_html=True)   
                                else:
                                    st.markdown(
                                        "<h1 style='color:green;'>This patient is predicted to survive.</h3>", 
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        f"<p style='font-size:26px;'>💚 <b>Probability of SURVIVAL:</b> {probability[0][1] * 100:.2f}%</p>", 
                                        unsafe_allow_html=True)

                                #next steps recommendation
                                st.subheader("\nNext Steps:")
                                st.write("""
                                    Based on your prediction, we recommend discussing these results with your healthcare provider.
                                    You can schedule an appointment with your oncologist or visit support groups to further explore options.
                                """)

                                #uncertainty and Limitations
                                st.subheader("Disclaimer:")
                                st.write("""
                                    This prediction is based on a machine learning model trained on clinical data. While it may provide valuable insights,
                                    it is important to consult with a healthcare provider for a more personalized assessment.
                                """)
                            
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
                                
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader('Prediction Probability Distribution')
                            fig1 = plot_probability_distribution(probability, prediction)
                            st.pyplot(fig1)
                            
        except Exception as e:
            st.error(f"Error: {e}")
            
# Second page# Second page# Second page
elif page == "Custom Model Creation": 
    st.title('Build your own model!')
    st.write("Below is the pre-processed dataset used to create models to predict patient survivability:")
    st.dataframe(data_model_build)  
    num_rows, num_columns = data_model_build.shape

    st.write(f"**Number of Patient Data:** {num_rows}")
    st.write(f"**Number of Features:** {num_columns}")
    
    st.write("\n")
    st.subheader("Step 1: Feature Selection")
    
    # Preprocess the data
    X, y = preprocess_data_for_survival(data_model_build)
    
    # Slider for p-value threshold
    p_value_threshold = st.slider(
        "Select the p-value threshold for feature selection", 
        min_value=0.01, 
        max_value=0.1, 
        value=0.05, 
        step=0.01
    )

    significant_features = perform_t_test(X, y, p_value_threshold)

    significant_df = pd.DataFrame(significant_features, columns=["Feature"])
        
    table_col, count_col = st.columns(2)

    with table_col:
        st.subheader("Significant Features")
        st.dataframe(significant_df)

    with count_col:
        st.subheader("Feature Count")
        st.write(f"Count of Significant Features: {len(significant_features)}")

        st.subheader(r"\*Optional\*" , help="Recursive Feature Elimination (RFE) can be used to further select features based on the importance of each feature. - It might take some time to perform RFE.")

        rfe_checkbox = st.checkbox("Perform Recursive Feature Elimination (RFE)")

        if rfe_checkbox:  # Let user pick either SVM or RF
            estimator_choice = st.selectbox(
                "Select the Estimator for RFE (Random Forest or SVM)", 
                ["Random Forest", "SVM"]
            )

            num_features = st.slider(
                "Select the number of features to retain",
                min_value=1,
                max_value=25,
                value=10,
                step=1
            )

            rfe_button = st.button("Perform RFE")

            if rfe_button:
                if estimator_choice == "Random Forest":
                    estimator = RandomForestClassifier(random_state=42)
                elif estimator_choice == "SVM":
                    estimator = SVC(kernel="linear")  # Ensure linear kernel for SVC

                def perform_rfe_with_estimator(X_train, y_train, estimator, num_features):
                    rfe = RFE(estimator=estimator, n_features_to_select=num_features)
                    rfe.fit(X_train, y_train)
                    return X_train.columns[rfe.support_]

                if len(significant_features) > 0:
                    X_train_rfe = X[significant_features]
                    selected_features_rfe = perform_rfe_with_estimator(X_train_rfe, y, estimator, num_features)

                    st.subheader(f"Features Selected by RFE ({estimator_choice})")
                    st.write(f"Selected Features: {selected_features_rfe}")
                    
    st.write("\n")
    st.subheader("Step 2: Train Your Model")


    model_choice = st.multiselect(
        "Select the models to train", 
        ["SVC", "Random Forest"]
    )
    
    test_size_slider = st.slider(
        "Select Test Size (Proportion)",
        min_value=0.1,  
        max_value=0.5,  
        value=0.3,    
        step=0.05  
    )

    X_train, X_test, y_train, y_test = train_test_split(X[significant_features], y, test_size=test_size_slider, random_state=42)
    
    train_button = st.button("Train Models")

        
    if train_button:
        model_results = {}

        #train svc
        if "SVC" in model_choice:
            svc_model = SVC(kernel='linear', random_state=42, probability=True)
            svc_model.fit(X_train[significant_features], y_train)
            svc_accuracy = accuracy_score(y_test, svc_model.predict(X_test[significant_features]))
            model_results["SVC"] = (svc_model, svc_accuracy)

        #train rfc
        if "Random Forest" in model_choice:
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train[significant_features], y_train)
            rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test[significant_features]))
            model_results["Random Forest"] = (rf_model, rf_accuracy)

        #results
        for model_name, (model, accuracy) in model_results.items():
            st.write(f"Model trained! {model_name} Accuracy: {accuracy:.4f}")

        #roc curve
        st.subheader("ROC Curve Comparison")
        plt.figure(figsize=(10, 6))

        if "SVC" in model_results:
            fpr_svc, tpr_svc, _ = roc_curve(y_test, svc_model.predict_proba(X_test[significant_features])[:, 1])
            plt.plot(fpr_svc, tpr_svc, label="SVC ROC", color='blue')

        if "Random Forest" in model_results:
            fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test[significant_features])[:, 1])
            plt.plot(fpr_rf, tpr_rf, label="Random Forest ROC", color='green')

        plt.plot([0, 1], [0, 1], linestyle='--', color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        st.pyplot(plt)

        #confusion matrix
        st.subheader("Confusion Matrix Comparison")
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        if "SVC" in model_results:
            svc_cm = confusion_matrix(y_test, svc_model.predict(X_test[significant_features]))
            ConfusionMatrixDisplay(confusion_matrix=svc_cm).plot(ax=ax[0])
            ax[0].set_title('SVC Confusion Matrix')

        if "Random Forest" in model_results:
            rf_cm = confusion_matrix(y_test, rf_model.predict(X_test[significant_features]))
            ConfusionMatrixDisplay(confusion_matrix=rf_cm).plot(ax=ax[1])
            ax[1].set_title('Random Forest Confusion Matrix')

        st.pyplot(fig)

        for model_name, (model, _) in model_results.items():
            if model_name == "SVC":
                if hasattr(model, 'coef_'):
                    feature_importances = np.abs(model.coef_[0])
            elif model_name == "Random Forest":
                feature_importances = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                "Feature": significant_features,
                "Importance": feature_importances
            })

            top_10_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)
            st.subheader(f"Top 10 Features Ranked by Importance ({model_name})")
            st.dataframe(top_10_features)
            
            if model_name == "SVC":
                model_filename = "svc_model.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()  # encode to base64
                href = f'<a href="data:file/pkl;base64,{b64}" download="svc_model.pkl">Download SVC Model</a>'
                st.markdown(href, unsafe_allow_html=True)

            elif model_name == "Random Forest":
                model_filename = "rf_model.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()  # encode to base64
                href = f'<a href="data:file/pkl;base64,{b64}" download="rf_model.pkl">Download Random Forest Model</a>'
                st.markdown(href, unsafe_allow_html=True)


if page == "Test Your Model":
    st.title('Test Your Own Breast Cancer Survival Model')
    st.markdown('*****Upload your model and a CSV file to test predictions.*****')

    uploaded_model = st.file_uploader("Upload your model file (.pkl)", type=['pkl', 'joblib'])
    if uploaded_model:
        model = joblib.load(uploaded_model)
        st.success("Model loaded successfully!")

    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

    if uploaded_file and 'model' in locals():
        try:
            data = pd.read_csv(uploaded_file)
            
            if data.empty:
                st.error("Error: The uploaded file is empty. Please upload a valid CSV file with data.")
            else:
                if data.shape[0] > data.shape[1]:
                    data = data.T 
                    data.columns = data.iloc[0]  
                    data = data[1:]  
                    st.write("Data was transposed. Now it's back to the original form.")
                    data.reset_index(inplace=True)
                    data.rename(columns={'index': 'age_at_diagnosis'}, inplace=True)  
                
                st.write("Original Data:", data.head())

                if st.button('Predict'):
                    with st.spinner('Making prediction...'):
                        try:
                            prediction, probability, _ = make_prediction_all_features(model, data)

                            if prediction is not None:  # Check if prediction was made
                                if probability is not None:  # Check if model can predict probabilities
                                    st.subheader('Prediction Results')
                                    if prediction[0] == 0:
                                        st.markdown(
                                            "<h1 style='color:red;'>This patient is predicted to NOT survive.</h3>", 
                                            unsafe_allow_html=True)
                                        st.markdown(
                                            f"<p style='font-size:26px;'>🛑 <b>Probability of NOT SURVIVING:</b> {probability[0][0] * 100:.2f}%</p>", 
                                            unsafe_allow_html=True)   
                                    else:
                                        st.markdown(
                                            "<h1 style='color:green;'>This patient is predicted to survive.</h3>", 
                                            unsafe_allow_html=True)
                                        st.markdown(
                                            f"<p style='font-size:26px;'>💚 <b>Probability of SURVIVAL:</b> {probability[0][1] * 100:.2f}%</p>", 
                                            unsafe_allow_html=True)

                                else:
                                    st.write("This model does not support probability predictions.")
                                    st.write(f"Prediction: {'Survive' if prediction[0] == 1 else 'Not Survive'}")
                            else:
                                st.error("Failed to make a prediction due to feature mismatch or other issues.")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

        except Exception as e:
            st.error(f"Error: {e}")
