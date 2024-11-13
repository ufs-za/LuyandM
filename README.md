# Purchase Intention Predictor

This Streamlit app predicts a customer's purchase intention based on various factors. It uses a pre-trained Random Forest model to make predictions and visualizes the results using an interactive chart.

## Features

- Interactive sliders and dropdowns for input
- Chart for visualizing prediction probabilities

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/purchase-intention-predictor.git
   cd purchase-intention-predictor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the `random_forest_model.pkl` file in the same directory as the `app.py` file.

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

The app will open in your default web browser. Adjust the sliders and dropdowns to see real-time prediction updates.

## Deployment

This app is ready for deployment on GitHub Codespaces. To use it in Codespaces:

1. Open the repository in GitHub Codespaces
2. Install the requirements: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
4. Open the app in the browser when prompted
