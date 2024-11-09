import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go
from plotly.colors import n_colors
import os

# Set page configuration
st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛒", layout="wide")

# Custom CSS for yellow theme with black text
st.markdown("""
    <style>
    .main, .stApp {
        background-color: #fff9e6;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #fff3cc;
    }
    .stSlider > div > div {
        color: #000000;
    }
    .stSelectbox > div {
        font-size: 16px;
        color: #000000;
    }
    .stPlotlyChart {
        border: 2px solid #ffd700;
        padding: 5px;
        border-radius: 10px;
        background-color: #fff3cc;
    }
    h1, h2, h3, h4, h5 {
        color: #b38600;
    }
    .stAlert {
        background-color: #fff3cc;
        color: #000000;
    }
    .stSpinner > div > div {
        border-top-color: #ffd700 !important;
    }
    .stButton>button {
        background-color: #ffd700;
        color: #000000;
        font-weight: bold;
        border: 2px solid #b38600;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #ffed4a;
        border-color: #ffd700;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("💡 Purchase Intention Predictor")

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("🚨 Model file not found. Please ensure 'random_forest_model.pkl' is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None

with st.spinner("Loading model..."):
    model = load_model()

if model:
    st.success("🎉 Model loaded successfully!")
else:
    st.error("Failed to load the model. Please check the model file path.")

# Slider labels with humor
ppq1_labels = {
    1: "🤢 Very poor", 2: "😬 Poor", 3: "😐 Moderate", 4: "😄 Good", 5: "🤩 Very good"
}

pv3_labels = {
    1: "💸 Cannot save", 2: "💸 Rarely save", 3: "🤔 Sometimes save", 4: "💵 Can save", 5: "💰 Always save"
}

pv2_labels = {
    1: "🚫 Unaffordable", 2: "😓 Barely affordable", 3: "🤷 Fairly affordable", 4: "💳 Affordable", 5: "🤑 Very affordable"
}

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Customize Your Customer Profile:")

    # Sliders with labels and descriptions
    ppq1 = st.slider("Perceived Product Quality", 1, 5, 3, key="ppq1")
    st.write(f"**Selected:** {ppq1_labels[ppq1]}")

    pv3 = st.slider("Ability to Save Money at Local Store", 1, 5, 3, key="pv3")
    st.write(f"**Selected:** {pv3_labels[pv3]}")

    pv2 = st.slider("Affordability of Product Offering", 1, 5, 3, key="pv2")
    st.write(f"**Selected:** {pv2_labels[pv2]}")

    # Age and Gender dropdowns
    age = st.selectbox(
        "Select Age Group", 
        [1, 2, 3, 4, 5], 
        format_func=lambda x: {1: "18-22 (Gen Z)", 2: "23-28 (Millennials)", 3: "29-35 (Young Professionals)", 4: "36-49 (Prime Buyers)", 5: "50-65 (Boomers)"}[x]
    )
    gender = st.selectbox(
        "Select Gender", 
        [1, 2, 3], 
        format_func=lambda x: {1: "Male", 2: "Female", 3: "Prefer not to say"}[x]
    )

    # Add prediction button
    predict_button = st.button("🔮 Make Prediction")

with col2:
    st.header("🔮 Prediction Results")

    # Only show prediction when button is clicked
    if predict_button and model is not None:
        # Prepare features (as a 2D array)
        features = np.array([[ppq1, pv3, pv2, age, gender]])

        with st.spinner("Updating prediction..."):
            try:
                prediction = model.predict(features)
                prediction_probabilities = model.predict_proba(features)[0]

                # Displaying the prediction with thoughtful explanation
                prediction_label = ""
                if prediction == 0:
                    prediction_label = "😔 Prediction: **No intention** to purchase."
                elif prediction == 1:
                    prediction_label = "✅ Prediction: **Strong intention** to purchase!"
                else:
                    prediction_label = "🤔 Prediction: **Neutral intention** to purchase."

                st.success(prediction_label)

                # Create pie chart using Plotly
                labels = ['No intention', 'Strong intention', 'Neutral intention']
                values = prediction_probabilities.round(4) * 100  # Display in percentage

                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,  # Creates a donut chart
                        marker=dict(
                            colors=['#fff3cc', '#ffd700', '#b38600'],
                            line=dict(color='#000000', width=1.5)
                        ),
                        textfont=dict(color='#000000', size=14),
                        hovertemplate="<b>%{label}</b><br>" +
                                    "Probability: %{value:.1f}%<br>" +
                                    "<extra></extra>",
                        texttemplate="%{label}<br>%{value:.1f}%"
                    )
                ])

                fig.update_layout(
                    title=dict(
                        text="🔎 Probability Distribution of Purchase Intention",
                        font=dict(color='#000000', size=16),
                        y=0.95
                    ),
                    plot_bgcolor='rgba(255,249,230,0)',
                    paper_bgcolor='rgba(255,249,230,0)',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                    annotations=[
                        dict(
                            text="Purchase<br>Intention",
                            x=0.5,
                            y=0.5,
                            font=dict(size=14, color='#000000'),
                            showarrow=False
                        )
                    ]
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❗ Error during prediction: {e}")
    elif not predict_button:
        st.info("Click the 'Make Prediction' button to see results!")
    else:
        st.warning("⚠️ Model is not loaded correctly.")