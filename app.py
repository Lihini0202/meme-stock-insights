import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pickle
import gdown
import zipfile
import xgboost as xgb

# Set page config
st.set_page_config(page_title="Meme Stock Insights", layout="wide")

# Title and description
st.title("📈 Meme Stock Insights Dashboard")
st.markdown("""
This app provides insights into meme stocks based on synthetic and processed data. 
Explore datasets, visualizations, trading signals, and model insights.
Meme images and processed data are downloaded from Google Drive due to size constraints.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", ["Datasets", "Visualizations", "Meme Gallery", "Trading Signals", "Model Insights"])

# Function to download and extract meme_images from Google Drive
@st.cache_resource
def download_meme_images():
    drive_link = "https://drive.google.com/file/d/1k5pODJE8e3omLmJotJZFfXrCZcM4XzoO"
    file_id = drive_link.split('/d/')[1].split('/')[0]
    output_zip = "meme_images.zip"
    output_folder = "meme_images"
    
    # Download zip file
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip, quiet=False)
    
    # Verify and extract zip
    if os.path.exists(output_zip):
        try:
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            st.success(f"Extracted images to {output_folder}/")
        except zipfile.BadZipFile:
            st.error("Invalid or corrupted zip file. Please verify the Google Drive link and zip integrity.")
            return None
    else:
        st.error("Failed to download zip file from Google Drive.")
        return None
    
    return output_folder

# Function to download processed_meme_data.pkl from Google Drive
@st.cache_resource
def download_processed_data():
    drive_link = "https://drive.google.com/file/d/1DSpPKfF_CAzpOaPTRATkC9YjWR_bFND-"
    file_id = drive_link.split('/d/')[1].split('/')[0]
    output_file = "data/processed_meme_data.pkl"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Download pickle file
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
    
    if os.path.exists(output_file):
        st.success(f"Downloaded processed_meme_data.pkl to {output_file}")
        with open(output_file, 'rb') as f:
            processed_data = pickle.load(f)
        return processed_data
    else:
        st.error("Failed to download processed_meme_data.pkl from Google Drive.")
        return None

# Load data
@st.cache_data
def load_data():
    try:
        synthetic_df = pd.read_csv('data/synthetic_meme_dataset_with_images.csv')
        signals_df = pd.read_csv('data/tradeable_signals.csv')
        mapped_df = pd.read_csv('data/mapped_meme_data.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading CSV files: {e}. Ensure data files are in the 'data/' folder.")
        return None, None, None, None
    
    processed_data = download_processed_data()
    
    return synthetic_df, signals_df, mapped_df, processed_data

# Download images and processed data
image_folder = download_meme_images()
if image_folder is None:
    image_folder = "meme_images"  # Fallback to local folder if download fails

synthetic_df, signals_df, mapped_df, processed_data = load_data()
if synthetic_df is None or processed_data is None:
    st.stop()

if page == "Datasets":
    st.header("📊 Datasets Overview")
    
    st.subheader("Synthetic Meme Dataset")
    st.dataframe(synthetic_df.head(10))
    
    st.subheader("Mapped Meme Data")
    st.dataframe(mapped_df.head(10))
    
    st.subheader("Tradeable Signals")
    st.dataframe(signals_df.head(10))
    
    st.subheader("Processed Meme Data (from Pickle)")
    if isinstance(processed_data, pd.DataFrame):
        st.dataframe(processed_data.head(10))
    else:
        st.write("Processed data (non-DataFrame, likely model):", type(processed_data))

elif page == "Visualizations":
    st.header("📉 Visualizations")
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution (Synthetic Data)")
    if 'sentiment' in synthetic_df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='sentiment', data=synthetic_df, ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
    
    # Price Change Histogram
    st.subheader("Price Change Histogram")
    if 'price_change' in synthetic_df.columns:
        fig, ax = plt.subplots()
        sns.histplot(synthetic_df['price_change'], kde=True, ax=ax)
        ax.set_title("Price Change Distribution")
        st.pyplot(fig)
    
    # Virality vs. Trading Volume
    st.subheader("Virality vs. Trading Volume")
    if 'virality' in synthetic_df.columns and 'volume' in synthetic_df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(x='virality', y='volume', hue='sentiment', size='price_change', data=synthetic_df, ax=ax)
        ax.set_title("Virality vs. Volume by Sentiment")
        st.pyplot(fig)

elif page == "Meme Gallery":
    st.header("🖼️ Meme Gallery")
    st.markdown("Displaying sample memes from the dataset, sourced from Google Drive.")
    
    if 'image_path' in synthetic_df.columns:
        selected_ticker = st.selectbox("Filter by Ticker", options=sorted(synthetic_df['ticker'].unique()))
        filtered_df = synthetic_df[synthetic_df['ticker'] == selected_ticker]
        
        for index, row in filtered_df.iterrows():
            img_path = os.path.join(image_folder, row['image_path'].split('/')[-1])
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption=row['caption'], use_column_width=True)
                st.write(f"Sentiment: {row['sentiment']}, Virality: {row['virality']}")
            else:
                st.warning(f"Image not found: {img_path}. Ensure Google Drive link is correct and images are extracted.")
    else:
        st.error("No 'image_path' column found in synthetic dataset.")

elif page == "Trading Signals":
    st.header("💹 Trading Signals")
    st.dataframe(signals_df)
    
    # Signal Summary
    st.subheader("Signal Summary")
    if 'signal' in signals_df.columns:
        signal_counts = signals_df['signal'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=signal_counts.index, y=signal_counts.values, ax=ax)
        ax.set_title("Trading Signal Distribution")
        st.pyplot(fig)

elif page == "Model Insights":
    st.header("🧠 Model Insights")
    st.markdown("Insights from the processed_meme_data.pkl (assumed to be an XGBoost model or features).")
    
    if isinstance(processed_data, pd.DataFrame):
        st.subheader("Processed Features")
        st.dataframe(processed_data.head())
    else:
        # Assume it's an XGBoost model
        try:
            st.subheader("XGBoost Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(processed_data, ax=ax, max_num_features=10)
            ax.set_title("Top 10 Feature Importance")
            st.pyplot(fig)
            
            # Example: Predict on sample data (modify based on your features)
            st.subheader("Sample Prediction")
            if 'virality' in synthetic_df.columns and 'sentiment' in synthetic_df.columns:
                sample_data = synthetic_df[['virality', 'sentiment']].head(1)
                if isinstance(processed_data, xgb.Booster):
                    dmatrix = xgb.DMatrix(sample_data)
                    prediction = processed_data.predict(dmatrix)
                    st.write("Sample prediction (first row):", prediction)
                else:
                    st.write("Model is not an XGBoost Booster. Skipping prediction.")
        except Exception as e:
            st.error(f"Error plotting model insights or predicting: {e}. Ensure processed_meme_data.pkl is an XGBoost model.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit. Data from meme-stock-insights project. Images and model sourced from Google Drive.")
