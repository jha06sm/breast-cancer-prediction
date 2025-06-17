import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Set page config
st.set_page_config(
    page_title="Breast Cancer Detector",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    # Load your actual model and scaler here
    # For this example, we'll create a dummy model with 30 features
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=569, n_features=30, random_state=42)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Create realistic feature names matching your dataset
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se',
        'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Main app
st.title("Breast Cancer Detection App")
st.write("""
This app predicts whether a breast tumor is **malignant** or **benign** using machine learning.
""")

# Create tabs
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Single Tumor Prediction")
    
    # Create input sections for different feature groups
    st.subheader("Mean Features")
    cols = st.columns(3)
    inputs = {}
    
    # Mean features (first 10)
    for i, feature in enumerate(feature_names[:10]):
        with cols[i % 3]:
            inputs[feature] = st.number_input(
                label=feature,
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=0.1,
                key=f"mean_{feature}"
            )
    
    st.subheader("Standard Error Features")
    cols = st.columns(3)
    # SE features (next 10)
    for i, feature in enumerate(feature_names[10:20]):
        with cols[i % 3]:
            inputs[feature] = st.number_input(
                label=feature,
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=0.1,
                key=f"se_{feature}"
            )
    
    st.subheader("Worst Features")
    cols = st.columns(3)
    # Worst features (last 10)
    for i, feature in enumerate(feature_names[20:]):
        with cols[i % 3]:
            inputs[feature] = st.number_input(
                label=feature,
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=0.1,
                key=f"worst_{feature}"
            )
    
    # Prediction button
    if st.button("Predict Single Tumor"):
        # Prepare input data with all 30 features
        input_data = np.array([inputs[feat] for feat in feature_names]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)
        
        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"**Malignant** (confidence: {proba[0][1]:.1%})")
        else:
            st.success(f"**Benign** (confidence: {proba[0][0]:.1%})")

with tab2:
    st.header("Batch Prediction")
    
    st.warning("For batch prediction, your CSV must contain all 30 features in the correct order.")
    st.write("Required features:", ", ".join(feature_names))
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file with tumor features",
        type=["csv"],
        help="File should contain all 30 features used to train the model"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Check if all features are present
            if set(feature_names).issubset(set(df.columns)):
                if st.button("Predict Batch"):
                    # Preprocess and predict
                    X = df[feature_names]
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    probas = model.predict_proba(X_scaled)
                    
                    # Add predictions to dataframe
                    df['Prediction'] = ['Malignant' if p == 1 else 'Benign' for p in predictions]
                    df['Confidence'] = [probas[i][1] if p == 1 else probas[i][0] for i, p in enumerate(predictions)]
                    
                    # Show results
                    st.subheader("Prediction Results")
                    st.dataframe(df)
                    
                    # Download button
                    st.download_button(
                        label="Download Predictions",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name='breast_cancer_predictions.csv',
                        mime='text/csv'
                    )
            else:
                missing = set(feature_names) - set(df.columns)
                st.error(f"Missing required features: {', '.join(missing)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("""
This app uses a Random Forest classifier trained on the Wisconsin Breast Cancer Dataset.

**Note:** This is a demo app. For actual medical diagnosis, consult a healthcare professional.
""")

# Add some styling
st.markdown("""
<style>
    .stNumberInput, .stTextInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)