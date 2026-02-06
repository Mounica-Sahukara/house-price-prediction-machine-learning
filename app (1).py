import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("indian_housing_prices_final.csv")

# Selecting features and target variable
X = df[['Median_Income', 'House_Age', 'Avg_Rooms', 'Avg_Bedrooms', 'Population', 'Avg_Occupancy']]
y = df['Price_INR']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

def predict_price(medinc, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy):
    new_house = np.array([[medinc, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy]])
    predicted_price = model.predict(new_house)[0]
    return f"🏡 Estimated Price: ₹{round(predicted_price, 2)} 🏠"

# Enhanced Gradio UI with emojis and better styling
css = """
    body {background-color: #e3f2fd; font-family: Arial, sans-serif;}
    .gradio-container {max-width: 650px; margin: auto; text-align: center; background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);}
    h1 {color: #1565c0; font-size: 28px;}
    .gradio-interface .description {color: #1e88e5; font-size: 16px; font-weight: bold; margin-bottom: 15px;}
    input {border-radius: 8px; padding: 10px; border: 1px solid #90caf9; width: 100%; margin-bottom: 10px;}
    button {background-color: #1e88e5; color: white; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; font-size: 16px;}
    button:hover {background-color: #1565c0;}
    .footer {margin-top: 20px; font-size: 14px; color: #555;}
"""

inputs = [
    gr.Number(label="💰 Median Income "),
    gr.Number(label="🏠 House Age" ),
    gr.Number(label="🛏️ Average Rooms"),
    gr.Number(label="🛋️ Average Bedrooms"),
    gr.Number(label="🌆 Population "),
    gr.Number(label="👨‍👩‍👧‍👦 Average Occupancy")
]

interface = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=gr.Textbox(label="✨ Predicted Price (INR) ✨"),
    title="🏡 Housing Price Prediction 🏠",
    theme="compact",
    css=css
)

interface.launch()