import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# -------------------------
# Load model (do NOT cache)
# -------------------------
model = tf.keras.models.load_model('model.h5')

# Cache pickle objects (safe)
@st.cache_data
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

label_encoder_gender = load_pickle('label_encoder_gender.pkl')
onehot_encoder_geo = load_pickle('onehot_encoder_geo.pkl')
scaler = load_pickle('scaler.pkl')

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title('💼 Customer Churn Prediction')

st.subheader("Enter Customer Details:")

with st.form("churn_form"):
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)
    tenure = st.slider('Tenure (Years)', 0, 10, 3)
    balance = st.number_input('Balance', min_value=0.0, step=100.0, format="%.2f")
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, step=1, value=600)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=100.0, format="%.2f")
    
    submit_button = st.form_submit_button("Predict Churn")

if submit_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine features
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display input summary
    st.subheader("Customer Input Summary:")
    st.table(input_data)

    # Display prediction result
    st.subheader("Prediction Result:")
    if prediction_proba > 0.5:
        st.markdown(f"<h2 style='color:red'>Churn Probability: {prediction_proba:.2f}</h2>", unsafe_allow_html=True)
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.markdown(f"<h2 style='color:green'>Churn Probability: {prediction_proba:.2f}</h2>", unsafe_allow_html=True)
        st.success("✅ The customer is not likely to churn.")
