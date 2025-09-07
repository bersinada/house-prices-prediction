# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Data Analysis", "Model Info", "About"])

with tab1:
    # Page configuration
    st.set_page_config(
        page_title="House Prices Prediction",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #3e4a61;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .prediction-box {
            background-color: #252530;
            padding: 2rem;
            border-radius: 1rem;
            border: 2px solid #FF4B4B;
            text-align: center;
            margin: 2rem 0;
            color: #FF4B4B;
        }
        /* Tab font size */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            font-size: 18px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #FF4B4B;
            color: white;
        }
        .stTabs [aria-selected="false"] {
            background-color: #f0f2f6;
            color: #FF4B4B;
        }
    </style>
    """, unsafe_allow_html=True)


    @st.cache_data
    def load_model():
        """Load the trained model and encoders"""
        try:
            model = joblib.load('models/trained_models/housing_model.pkl')
            encoders = joblib.load('models/trained_models/label_encoders.pkl')
            return model, encoders
        except FileNotFoundError:
            st.error("Model files not found. Please run the model development notebook first.")
            return None, None

    def main():
        # Main header
        st.markdown('<h1 class="main-header">🏠 House Prices Prediction</h1>', unsafe_allow_html=True)

        # Load model
        model, encoders = load_model()

        if model is None:
            st.stop()

        # Sidebar - Prediction form
        st.sidebar.header("🏠 Enter House Information")

        # Form fields
        longitude = st.sidebar.slider("Longitude", min_value=-124.0, max_value=-114.0, value=-118.0, step=0.1)
        latitude = st.sidebar.slider("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.1)
        housing_median_age = st.sidebar.slider("Housing Median Age", min_value=1, max_value=52, value=20 )
        total_rooms = st.sidebar.slider("Total Rooms", min_value=2, max_value=40000, value=2000)
        total_bedrooms = st.sidebar.slider("Total Bedrooms", min_value=1, max_value=6500, value=400)
        population = st.sidebar.slider("Population", min_value=3, max_value=36000, value=1500)
        households = st.sidebar.slider("Households", min_value=1, max_value=6100, value=500)
        median_income = st.sidebar.slider("Median Income", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
        ocean_proximity = st.sidebar.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

        # Prediction button
        if st.sidebar.button("🔮 Predict House Price", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                'longitude': [longitude],
                'latitude': [latitude],
                'housing_median_age': [housing_median_age],
                'total_rooms': [total_rooms],
                'total_bedrooms': [total_bedrooms],
                'population': [population],
                'households': [households],
                'median_income': [median_income],
                'ocean_proximity': [ocean_proximity]
            })
            
            # Create engineered features
            input_data['rooms_per_household'] = input_data['total_rooms'] / input_data['households']
            input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms']
            input_data['population_per_household'] = input_data['population'] / input_data['households']

            # Create income category
            input_data['income_cat'] = pd.cut(input_data['median_income'],
            bins=[0, 1.5, 3.0, 4.5, 6.0, 15.0],
            labels=['Low', 'Low-Medium', 'Medium', 'Medium-High', 'High'])

            # Encode categorical variables
            input_data['ocean_proximity'] = encoders['ocean_proximity'].transform(input_data['ocean_proximity'])
            input_data['income_cat'] = encoders['income_cat'].transform(input_data['income_cat'])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Display result
            st.markdown(f"""
            <div class="prediction-box">
                <h2>💰 Predicted House Price</h2>
                <h1 style="color: #f4f4f4; font-size: 3rem;">${prediction:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Detailed information
            st.subheader("House Characteristics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Longitude", f"{longitude:.1f}")
                st.metric("Latitude", f"{latitude:.1f}")

            with col2: 
                st.metric("Housing Age", f"{housing_median_age} years")
                st.metric("Total Rooms", f"{total_rooms:,}")

            with col3:
                st.metric("Total Bedrooms", f"{total_bedrooms:,}")
                st.metric("Population", f"{population:,}")

            with col4:
                st.metric("Households", f"{households:,}")
                st.metric("Median Income", f"${median_income:,.1f}")
                st.metric("Ocean Proximity", ocean_proximity)



with tab2:
    st.header("Data Analysis")

    # Load sample data for visualization
    @st.cache_data
    def load_sample_data():
        return pd.read_csv('data/raw/housing.csv')

    sample_data = load_sample_data()

    # Key statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", f"{len(sample_data):,}")
    with col2:
        st.metric("Average Price", f"${sample_data['median_house_value'].mean():,.0f}")
    with col3:
        st.metric("Price Range", f"${sample_data['median_house_value'].min():,.0f} - ${sample_data['median_house_value'].max():,.0f}")
    with col4:
        st.metric("Features", f"{sample_data.shape[1]}")

    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample_data['median_house_value'], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('House Price Distribution')
    st.pyplot(fig)

    # Correlation with income
    st.subheader("Income vs Price Relationship")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample_data['median_income'], sample_data['median_house_value'], alpha=0.6, color='red')
    ax.set_xlabel('Median Income')
    ax.set_ylabel('Price ($)')
    ax.set_title('Income vs Price Scatter Plot')
    st.pyplot(fig)

with tab3:
    st.header("Model Information")

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test R^2", "0.8050")
    with col2:
        st.metric("Test RMSE", "$50,544")
    with col3:
        st.metric("Test MAE", "$32,718")
    
    st.subheader("Model Details")
    st.write("""
    **Algorithm**: Random Forest Regressor
    **Training Samples**: 16,512
    **Test Samples**: 4,128
    **Features**: 13 (including engineerd features)
    **Training Time**: ~2 minutes
    """)
    
    st.subheader("Feature Importance")
    feature_importance = {
        'median_income': 0.5165,
        'population_per_household': 0.1334,
        'ocean_proximity': 0.0711,
        'latitude': 0.0642,
        'longitude': 0.0639
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    ax.barh(features, importance, color='red', alpha=0.7)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 5 Feature Importance')
    st.pyplot(fig)

with tab4:
    st.header("About This Project")

    st.write("""
        ## 🏠 House Prices Prediction System
        
        This web application predicts house prices in California using machine learning techniques.
        
        ### 📊 Dataset
        - **Source**: California Housing Dataset
        - **Samples**: 20,640 housing records
        - **Features**: 9 numerical + 1 categorical
        - **Target**: Median house value ($)
        
        ### 🧠 Model
        - **Algorithm**: Random Forest Regressor
        - **Performance**: 80.5% variance explained (R² = 0.8050)
        - **Features**: Includes engineered features like rooms_per_household
        
        ### 🛠️ Technologies
        - **Python**: Data processing and modeling
        - **Scikit-learn**: Machine learning
        - **Streamlit**: Web application
        - **Pandas**: Data manipulation
        - **Matplotlib**: Visualization
        
        ### 👨‍💻 Developer
        **Sinan** - Data Science Portfolio
        
        ### 📄 License
        This project is licensed under the MIT License.
        """)

if __name__ == "__main__":
    main()

import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())