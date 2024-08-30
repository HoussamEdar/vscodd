import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('C:/Users/PC/Documents/vscodd/rf.pkl')

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #6200EA;
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 10px 24px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #3700B3;
    }
    .container {
        padding: 2rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.title('Term Deposit Subscription Prediction')
st.write("Use this tool to predict whether a client will subscribe to a term deposit based on their profile.")

# App pages
page = st.sidebar.selectbox("Choose a page", ["Single Prediction", "Batch Prediction"])

if page == "Single Prediction":
    st.markdown("### Please enter the client details below:")

    # Layout using columns
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        job = st.selectbox('Job', ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                   "blue-collar","self-employed","retired","technician","services"])
        marital = st.selectbox('Marital Status', ["married","divorced","single"])
        education = st.selectbox('Education', ["unknown","secondary","primary","tertiary"])
        default = st.radio('Has Credit Default?', ['yes', 'no'])
        housing = st.radio('Has Housing Loan?', ['yes', 'no'])
        loan = st.radio('Has Personal Loan?', ['yes', 'no'])

    with col2:
        contact = st.selectbox('Preferred Contact Method', ["unknown","telephone","cellular"])
        month = st.selectbox('Last Contact Month', ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        day_of_week = st.number_input('Last Contact Day', min_value=1, max_value=31, value=15)
        duration = st.number_input('Call Duration (seconds)', min_value=0, value=180)
        campaign = st.slider('Number of Contacts During Campaign', min_value=1, max_value=50, value=1)
        previous = st.slider('Previous Contacts Before Campaign', min_value=0, max_value=10, value=0)
        poutcome = st.selectbox('Outcome of Previous Campaign', ["unknown","other","failure","success"])

    # Prediction logic
    if st.button('Predict'):
        with st.spinner('Analyzing the data...'):
            user_data = {
                'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'default': default,
                'housing': housing,
                'loan': loan,
                'contact': contact,
                'month': month,
                'day': day_of_week,
                'duration': duration,
                'campaign': campaign,
                'pdays': -1,  # Default value for single prediction
                'previous': previous,
                'poutcome': poutcome
            }

            # Convert user data to a DataFrame and apply the same preprocessing
            user_df = pd.DataFrame([user_data])
            user_df = pd.get_dummies(user_df)
            user_df = user_df.reindex(columns=model.feature_names_in_, fill_value=0)

            # Make the prediction
            prediction = model.predict(user_df)[0]

        # Display the result
        if prediction == 1:
            st.success('✔️ The client is likely to subscribe to a term deposit.')
        else:
            st.warning('❌ The client is unlikely to subscribe to a term deposit.')

elif page == "Batch Prediction":
    st.markdown("### Upload a CSV file for batch prediction:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Process the DataFrame (e.g., one-hot encoding)
        df_processed = pd.get_dummies(df)
        df_processed = df_processed.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make predictions
        df['y'] = model.predict(df_processed)

        # Filter the output columns, only keep those that exist in the DataFrame
        output_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                          'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
        df_output = df[[col for col in output_columns if col in df.columns]]

        # Display the DataFrame
        st.write("### Prediction Results")
        st.dataframe(df_output)

        # Provide download link
        st.download_button(
            label="Download Predictions",
            data=df_output.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv'
        )
