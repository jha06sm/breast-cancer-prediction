# breast-cancer-prediction
Breast Cancer Detection App

Overview
This Streamlit application predicts whether a breast tumor is malignant or benign using machine learning. The app provides two prediction modes:

Single Prediction: Enter individual tumor characteristics manually

Batch Prediction: Upload a CSV file containing multiple tumor records

Features
Interactive interface with real-time predictions

Supports all 30 diagnostic features from the Wisconsin Breast Cancer Dataset

Clear visualization of prediction results with confidence scores

Batch processing capability for multiple predictions

Responsive design that works on desktop and mobile devices

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
Install the required packages:

bash
pip install -r requirements.txt
Usage
Run the application:

bash
streamlit run app.py
Access the app in your browser at http://localhost:8501

Choose your prediction mode:

Single Prediction: Enter values for all 30 features

Batch Prediction: Upload a CSV file containing the required features

File Structure
text
breast-cancer-detection/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── README.md             # This documentation
├── data/                 # Sample data directory
│   └── sample_data.csv   # Example input file
└── models/               # Trained models directory
    └── model.pkl         # Serialized model file
Requirements
Python 3.8+

Streamlit

scikit-learn

pandas

numpy

Deployment
Streamlit Sharing
Push your code to a GitHub repository

Go to Streamlit Sharing

Connect your GitHub account

Select repository and specify app.py as the file to run

Other Platforms
The app can also be deployed on:

Heroku

AWS EC2

Google Cloud Run

Docker containers

Sample Data Format
For batch prediction, your CSV file should include these 30 columns (in any order):

text
radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
compactness_mean,concavity_mean,concave points_mean,symmetry_mean,
fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,
smoothness_se,compactness_se,concavity_se,concave points_se,
symmetry_se,fractal_dimension_se,radius_worst,texture_worst,
perimeter_worst,area_worst,smoothness_worst,compactness_worst,
concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst
Important Notes
This application is for demonstration purposes only

Always consult a medical professional for actual diagnoses

Model performance depends on the quality of training data

Contributing
Contributions are welcome! Please open an issue or submit a pull request.
