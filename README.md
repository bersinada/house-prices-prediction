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
├── data/                              # Data files
│   ├── raw/                           # Raw data
│   └── processed/                     # Processed data
├── notebooks/                         # Jupyter notebooks
│   └── 01_data_exploration.ipynb     # Data exploration
├── src/                              # Source code
│   ├── data/                         # Data processing modules
│   ├── models/                       # Model development
│   ├── visualization/                # Visualization
│   └── utils/                        # Utility functions
├── models/                           # Trained models
│   ├── trained_models/              # Model files
│   └── model_artifacts/             # Model artifacts
├── app/                             # Web application
│   └── streamlit_app.py            # Streamlit application
├── tests/                           # Test files
└── docs/                            # Additional documentation
```

## 🚀 Installation

### Requirements

- Python 3.8+
- pip or conda

### Steps

1. **Clone the repository:**

```bash
git clone <repository-url>
cd House Prices Prediction
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

The project uses a house prices dataset with the following features:

- **area**: House area (m²)
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **age**: House age
- **location**: Location (Center, City, Suburb, Rural)
- **garage**: Garage availability (0/1)
- **garden**: Garden availability (0/1)
- **pool**: Pool availability (0/1)
- **distance_to_city**: Distance to city center (km)
- **price**: House price (TL) - Target variable

## 🔧 Usage

### Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Model Training

```bash
python src/models/model_trainer.py
```

### Web Application

```bash
streamlit run app/streamlit_app.py
```

## 📈 Model Performance

Models tested within the project scope:

| Model             | RMSE | R² | MAE |
| ----------------- | ---- | --- | --- |
| Linear Regression | -    | -   | -   |
| Random Forest     | -    | -   | -   |
| Gradient Boosting | -    | -   | -   |
| SVR               | -    | -   | -   |

## 🛠️ Technologies

- **Python 3.8+**
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn**: Data visualization
- **Streamlit**: Web application
- **Jupyter**: Data exploration
- **Git**: Version control

This project is licensed under the MIT License.
