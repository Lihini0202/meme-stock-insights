import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:
    st.error("Seaborn is not installed. Please ensure 'seaborn' is in requirements.txt and installed.")
    st.stop()
from PIL import Image
import os
import pickle
import gdown
import zipfile
import xgboost as xgb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Function to download and extract meme_image from Google Drive
@st.cache_resource
def download_meme_images():
    # Replace with the correct file link for meme_image.zip
    drive_link = "https://drive.google.com/drive/folders/1rkxpBmT1PKw3KAyu37GOrgSKrx8esH8y?usp=drive_link"  # Update with new file link
    try:
        file_id = drive_link.split('/d/')[1].split('/')[0]
    except IndexError:
        st.error("Invalid Google Drive link for meme_image.zip. Please provide a valid file link.")
        logger.error("Invalid Google Drive link for meme_image.zip: %s", drive_link)
        return None
    output_zip = "meme_image.zip"
    output_folder = "meme_images"
    
    # Download zip file
    try:
        logger.info("Downloading meme_image.zip from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip, quiet=False)
    except Exception as e:
        st.error(f"Failed to download meme_image.zip: {e}")
        logger.error("Failed to download meme_image.zip: %s", str(e))
        return None
    
    # Verify and extract zip
    if os.path.exists(output_zip):
        try:
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            st.success(f"Extracted images to {output_folder}/")
            logger.info("Extracted meme_image.zip to %s", output_folder)
        except zipfile.BadZipFile:
            st.error("Invalid or corrupted zip file. Please verify the Google Drive link and zip integrity.")
            logger.error("Invalid or corrupted zip file: %s", output_zip)
            return None
    else:
        st.error("Failed to download zip file from Google Drive.")
        logger.error("Zip file not found: %s", output_zip)
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
    try:
        logger.info("Downloading processed_meme_data.pkl from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
    except Exception as e:
        st.error(f"Failed to download processed_meme_data.pkl: {e}")
        logger.error("Failed to download processed_meme_data.pkl: %s", str(e))
        return None
    
    if os.path.exists(output_file):
        st.success(f"Downloaded processed_meme_data.pkl to {output_file}")
        logger.info("Downloaded processed_meme_data.pkl to %s", output_file)
        with open(output_file, 'rb') as f:
            processed_data = pickle.load(f)
        return processed_data
    else:
        st.error("Failed to download processed_meme_data.pkl from Google Drive.")
        logger.error("Pickle file not found: %s", output_file)
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
        logger.error("Error loading CSV files: %s", str(e))
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
        # Verify image_path values match meme_0.png to meme_49.png
        expected_images = [f"meme_{i}.png" for i in range(50)]
        invalid_paths = synthetic_df[~synthetic_df['image_path'].isin(expected_images)]['image_path'].unique()
        if len(invalid_paths) > 0:
            st.warning(f"Invalid image paths found in synthetic dataset: {invalid_paths}. Expected: meme_0.png to meme_49.png")
            logger.warning("Invalid image paths: %s", invalid_paths)
        
        selected_ticker = st.selectbox("Filter by Ticker", options=sorted(synthetic_df['ticker'].unique()))
        filtered_df = synthetic_df[synthetic_df['ticker'] == selected_ticker]
        
        for index, row in filtered_df.iterrows():
            img_path = os.path.join(image_folder, row['image_path'])
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption=row['caption'], use_column_width=True)
                st.write(f"Sentiment: {row['sentiment']}, Virality: {row['virality']}")
            else:
                st.warning(f"Image not found: {img_path}. Ensure Google Drive link is correct and images are extracted.")
                logger.warning("Image not found: %s", img_path)
    else:
        st.error("No 'image_path' column found in synthetic dataset.")
        logger.error("No 'image_path' column in synthetic dataset")

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
                    logger.info("Sample prediction made: %s", prediction)
                else:
                    st.write("Model is not an XGBoost Booster. Skipping prediction.")
                    logger.info("Model is not an XGBoost Booster, type: %s", type(processed_data))
            else:
                st.write("Required columns for prediction (e.g., 'virality', 'sentiment') not found.")
                logger.warning("Required columns for prediction not found in synthetic_df")
        except Exception as e:
            st.error(f"Error plotting model insights or predicting: {e}")
            logger.error("Error in model insights: %s", str(e))

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit. Data from meme-stock-insights project. Images and model sourced from Google Drive.")
