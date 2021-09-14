import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st


import joblib
#Load the model to disk
model = joblib.load(r"model1.sav")
from PIL import Image
image1 = Image.open('pp.jpg')
st.sidebar.image(image1)
#from preprocessing import preprocess
def main():
    # Setting Application title
    st.title('Insurance Churn Prediction App')

    # Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional insurance company use case. 
     The application is functional for both online prediction and batch data prediction.. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application sidebar default
    image = Image.open('PyCoders.jpg')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        # Based on our optimal features selection
        feature_0 = st.number_input('Feature 0	', value=0)
        feature_1 = st.number_input('Feature 1	', value=0)
        feature_2 = st.number_input('Feature 2	', value=0)
        feature_3 = st.number_input('Feature 3	', value=0)
        feature_4 = st.number_input('Feature 4	', value=0)
        feature_5 = st.number_input('Feature 5	', value=0)
        feature_6 = st.number_input('Feature 6	', value=0)
        feature_7 = st.number_input('Feature 7  ', min_value=0, max_value=11, value=0)
        feature_8 = st.selectbox("Feature 8", ('0', '1', '2'))
        feature_9 = st.selectbox("Feature 9", ('0', '1', '2', '4'))
        feature_10 = st.selectbox("Feature 10", ('0', '1'))
        feature_11 = st.selectbox("Feature 11", ('0', '1'))
        feature_12 = st.selectbox("Feature 12", ('0', '1'))
        feature_13 = st.selectbox("Feature 13", ('0', '1'))
        feature_14 = st.number_input('Feature 14  ', min_value=0, max_value=11, value=0)
        feature_15 = st.selectbox("Feature 15  ", ('0', '1', '2', '3'))

        data = {
            'feature_0': feature_0,
            'feature_1': feature_1,
            'feature_2': feature_2,
            'feature_3': feature_3,
            'feature_4': feature_4,
            'feature_5': feature_5,
            'feature_6': feature_6,
            'feature_7': feature_7,
            'feature_8': feature_8,
            'feature_9': feature_9,
            'feature_10': feature_10,
            'feature_11': feature_11,
            'feature_12': feature_12,
            'feature_13': feature_13,
            'feature_14': feature_14,
            'feature_15': feature_15
        }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        # Preprocess inputs
        #preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(features_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the Company.')
            else:
                st.success('No, the customer is happy with Insurance Company.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            # Preprocess inputs
            #preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                # Get batch prediction
                prediction = model.predict(data)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the customer will terminate the Company.',
                                                       0: 'No, the customer is happy with Insurance Company.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)


if __name__ == '__main__':
    main()
