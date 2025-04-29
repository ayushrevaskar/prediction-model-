import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib

# Simple CSS for attractive UI
st.markdown("""
    <style>
    .title {
        color: #2E86C1;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        color: #34495E;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
    }
    .card {
        background-color: #F8F9F9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1A2526; /* Added dark text color */
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📱 Phone Recommender</div>', unsafe_allow_html=True)

# Load the CSV file
def load_data():
    try:
        df = pd.read_csv("Mobile_phone_price.csv")
    except FileNotFoundError:
        st.error("🚫 Error: 'Mobile_phone_price.csv' not found. Put it in the same folder as this script.")
        return None

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)

    # Check for required columns
    required_columns = ['Brand', 'Model', 'Storage', 'RAM', 'Screen_Size_inches', 'Camera_MP', 'Battery_Capacity_mAh', 'Price_']
    if not all(col in df.columns for col in required_columns):
        st.error("🚫 Error: CSV file is missing required columns.")
        return None

    # Clean data
    def clean_price(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '')
            try:
                return float(value)
            except ValueError:
                return None
        return float(value)

    def clean_camera(camera_str):
        if pd.isna(camera_str):
            return None
        match = re.match(r'(\d+\.?\d*)', str(camera_str))
        return float(match.group(0)) if match else None

    def clean_storage(storage):
        if pd.isna(storage):
            return None
        storage = storage.strip().upper()
        try:
            if 'GB' in storage:
                return float(storage.replace('GB', ''))
            elif 'TB' in storage:
                return float(storage.replace('TB', '')) * 1024
        except ValueError:
            return None
        return None

    def clean_ram(ram):
        if pd.isna(ram):
            return None
        ram = ram.strip().upper()
        try:
            return float(ram.replace('GB', ''))
        except ValueError:
            return None

    def clean_screen_size(screen):
        if pd.isna(screen):
            return None
        match = re.match(r'(\d+\.?\d*)', str(screen))
        return float(match.group(0)) if match else None

    # Apply cleaning
    df['Price'] = df['Price_'].apply(clean_price)
    df['Storage'] = df['Storage'].apply(clean_storage)
    df['RAM'] = df['RAM'].apply(clean_ram)
    df['Screen_Size'] = df['Screen_Size_inches'].apply(clean_screen_size)
    df['Camera'] = df['Camera_MP'].apply(clean_camera)
    df['Battery'] = df['Battery_Capacity_mAh'].apply(clean_price)  # Fixed: use clean_price

    # Keep only needed columns
    df = df[['Brand', 'Model', 'Price', 'Storage', 'RAM', 'Screen_Size', 'Camera', 'Battery']]

    # Remove rows with missing values
    df = df.dropna()
    if df.empty:
        st.error("🚫 Error: No valid data in CSV. Check if the data is correct.")
        return None

    return df

# Compute similarity between phones
def compute_similarity(df):
    try:
        features = ['Price', 'Storage', 'RAM', 'Screen_Size', 'Camera', 'Battery']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        similarity_matrix = cosine_similarity(scaled_features)
        return similarity_matrix
    except Exception:
        st.error("🚫 Error: Could not compute similarity. Check your CSV data.")
        return None

# Get recommendations
def get_recommendations(phone_index, df, similarity_matrix):
    try:
        sim_scores = list(enumerate(similarity_matrix[phone_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5 similar phones
        phone_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[phone_indices][['Brand', 'Model', 'Price', 'Storage', 'RAM', 'Screen_Size', 'Camera', 'Battery']].copy()
        recommendations['Similarity'] = [sim_scores[i][1] for i in range(len(sim_scores))]
        return recommendations
    except Exception:
        st.error("🚫 Error: Could not generate recommendations. Try another phone.")
        return None

# Load the .pkl model
def load_model():
    try:
        model = joblib.load('phone_price_model.pkl')
        return model
    except FileNotFoundError:
        st.error("🚫 Error: 'phone_price_model.pkl' not found. Put it in the same folder as this script.")
        return None

# Main app
def main():
    # Load data
    df = load_data()
    if df is None:
        return

    # Compute similarity
    similarity_matrix = compute_similarity(df)
    if similarity_matrix is None:
        return

    # Load model
    model = load_model()
    if model is None:
        return

    # Layout with two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Select a phone
        st.markdown('<div class="header">🎯 Pick a Phone</div>', unsafe_allow_html=True)
        phone_names = [f"{row['Brand']} {row['Model']}" for _, row in df.iterrows()]
        if not phone_names:
            st.error("🚫 No phones to show. Check your CSV file.")
            return

        selected_phone = st.selectbox("Choose a phone:", phone_names)
        selected_index = df.index[phone_names.index(selected_phone)]

        # Show recommendations
        if st.button("Show Similar Phones"):
            recommendations = get_recommendations(selected_index, df, similarity_matrix)
            if recommendations is not None:
                st.markdown('<div class="header">✅ Recommended Phones</div>', unsafe_allow_html=True)
                for _, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(
                            f"""
                            <div class="card">
                                <b>{row['Brand']} {row['Model']}</b><br>
                                Price: ${row['Price']:,.2f}<br>
                                Storage: {int(row['Storage'])} GB<br>
                                RAM: {int(row['RAM'])} GB<br>
                                Screen Size: {row['Screen_Size']:.2f} inches<br>
                                Camera: {int(row['Camera'])} MP<br>
                                Battery: {int(row['Battery'])} mAh<br>
                                Similarity: {row['Similarity']:.3f}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                st.success("Recommendations shown!")
            else:
                st.error("Could not show recommendations. Try another phone.")

    with col2:
        # Show selected phone details
        st.markdown('<div class="header">📋 Selected Phone</div>', unsafe_allow_html=True)
        selected_phone_data = df.loc[selected_index]
        with st.container():
            st.markdown(
                f"""
                <div class="card">
                    <b>{selected_phone_data['Brand']} {selected_phone_data['Model']}</b><br>
                    Price: ${selected_phone_data['Price']:,.2f}<br>
                    Storage: {int(selected_phone_data['Storage'])} GB<br>
                    RAM: {int(selected_phone_data['RAM'])} GB<br>
                    Screen Size: {selected_phone_data['Screen_Size']:.2f} inches<br>
                    Camera: {int(selected_phone_data['Camera'])} MP<br>
                    Battery: {int(selected_phone_data['Battery'])} mAh
                </div>
                """,
                unsafe_allow_html=True
            )

        # Predict price category
        features = [
            selected_phone_data['Storage'],
            selected_phone_data['RAM'],
            selected_phone_data['Screen_Size'],
            selected_phone_data['Camera'],
            selected_phone_data['Battery']
        ]
        features_array = np.array(features).reshape(1, -1)
        if model:
            prediction = model.predict(features_array)[0]
            price_category = "High Price" if prediction == 1 else "Low Price"
            st.markdown(f'<div class="card"><b>Price Category:</b> {price_category}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><b>Price Category:</b> Not available</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()