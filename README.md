# 🏠 House Prices Prediction

A comprehensive data science project that uses machine learning techniques to predict house prices based on various property features.

## 🎯 Project Goals

- **Data Analysis**: Analyze factors affecting house prices
- **Model Development**: Create high-accuracy prediction models
- **Visualization**: Interactive charts and dashboards
- **Web Application**: User-friendly prediction interface
- **Documentation**: Comprehensive project documentation

## 🏗️ Project Structure

```
House Prices Prediction/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore file
├── data/                              # Data files
│   ├── raw/                           # Raw data
│   │   └── housing.csv               # California Housing dataset
│   └── processed/                     # Processed data
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Data exploration and analysis
│   └── 02_model_development.ipynb   # Model training and evaluation
├── models/                           # Trained models
│   ├── trained_models/              # Model files
│   │   ├── housing_model.pkl        # Trained Random Forest model
│   │   └── label_encoders.pkl      # Label encoders for categorical data
│   └── model_artifacts/             # Model artifacts
└── app/                             # Web application
    └── streamlit_app.py            # Streamlit application (coming soon)
```

## 🚀 Installation

### Requirements

- Python 3.8+
- pip or conda

### Steps

1. **Clone the repository:**

```bash
git clone https://github.com/bersinada/house-prices-prediction
cd house-prices-prediction
```

2. **Create virtual environment:**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the **California Housing Dataset** with the following features:

- **longitude**: Block group longitude
- **latitude**: Block group latitude
- **housing_median_age**: Median house age in block group
- **total_rooms**: Total number of rooms in block group
- **total_bedrooms**: Total number of bedrooms in block group
- **population**: Block group population
- **households**: Number of households in block group
- **median_income**: Median income in block group
- **median_house_value**: Median house value ($) - Target variable
- **ocean_proximity**: Distance from ocean (categorical)

**Dataset Info:**

- **Total samples**: 20,640
- **Features**: 9 numerical + 1 categorical
- **Price range**: $14,999 - $500,001
- **Missing values**: 207 in total_bedrooms (handled)

## 🔧 Usage

### Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Model Development

```bash
jupyter notebook notebooks/02_model_development.ipynb
```

### Web Application

```bash
streamlit run app/streamlit_app.py
```

## 📈 Model Performance

Models tested within the project scope:

| Model             | Test R² | Test RMSE           | Test MAE  | Status |
| ----------------- | -------- | ------------------- | --------- | ------ |
| Random Forest     | 0.8050   | $50,544   | $32,718 | ✅ Best   |        |
| Gradient Boosting | 0.7781   | $53,928   | $36,963 | ✅ Good   |        |
| Ridge Regression  | 0.5906   | $73,246   | $51,520 | ⚠️ Fair |        |
| Lasso Regression  | 0.5877   | $73,500   | $51,567 | ⚠️ Fair |        |
| Linear Regression | 0.5876   | $73,517   | $51,570 | ⚠️ Fair |        |
| SVR               | -0.0487  | $117,229  | $87,344 | ❌ Poor   |        |

### 🏆 Best Model: Random Forest

- **Test R²**: 0.8050 (80.5% variance explained)
- **Test RMSE**: $50,544
- **Test MAE**: $32,718
- **Note**: Overfitting detected (Train R²: 0.97, Test R²: 0.81)

## 🛠️ Technologies

- **Python 3.8+**
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn**: Data visualization
- **Streamlit**: Web application
- **Jupyter**: Data exploration
- **Git**: Version control

This project is licensed under the MIT License.

## ⚠️ Known Limitations

- Price ceiling at $500,001 in the dataset (965 capped records)
- Mild overfitting observed (Train R² ≈ 0.97 vs Test R² ≈ 0.81) — acceptable for portfolio scope

## 🤝Acknowledgements

- Dataset: California Housing (derived from the 1990 U.S. Census)
- Libraries: scikit-learn, pandas, numpy, streamlit, matplotlib, seaborn

## ▶️ Run Locally

```bash
git clone https://github.com/bersinada/house-prices-prediction
cd house-prices-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
