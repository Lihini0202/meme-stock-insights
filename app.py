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

# --- GOOGLE DRIVE FILE IDs ---
# The File ID for  meme_images.zip
MEME_IMAGE_FILE_ID = "17y_b9nmOBx_ethy6tfv_Big8teFiD2OR" 
# The File ID for processed_meme_data.pkl
PROCESSED_DATA_FILE_ID = "1DSpPKfF_CAzpOaPTRATkC9YjWR_bFND-"
# --- END FILE IDs ---

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

## 📂 Data Download Functions

# Function to download and extract meme_images.zip from Google Drive
@st.cache_resource
def download_meme_images(file_id):
    output_zip = "meme_image.zip"
    output_folder = "meme_images"
    
    # Download zip file using the file ID
    try:
        logger.info("Downloading meme_images.zip from Google Drive with ID: %s", file_id)
        # Using the robust gdown format with the file ID
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip, quiet=False)
    except Exception as e:
        st.error(f"Failed to download meme_images.zip (ID: {file_id}): {e}")
        logger.error("Failed to download meme_images.zip: %s", str(e))
        return None
    
    # Verify and extract zip
    if os.path.exists(output_zip):
        try:
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            st.success(f"Extracted images to {output_folder}/")
            logger.info("Extracted meme_images.zip to %s", output_folder)
        except zipfile.BadZipFile:
            st.error("Invalid or corrupted zip file. Please verify the Google Drive ID and zip integrity.")
            logger.error("Invalid or corrupted zip file: %s", output_zip)
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during extraction: {e}")
            logger.error("Extraction error: %s", str(e))
            return None
    else:
        st.error("Failed to download zip file from Google Drive. Check File ID and sharing permissions.")
        logger.error("Zip file not found: %s", output_zip)
        return None
    
    return output_folder

# Function to download processed_meme_data.pkl from Google Drive
@st.cache_resource
def download_processed_data(file_id):
    output_file = "data/processed_meme_data.pkl"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Download pickle file using the file ID
    try:
        logger.info("Downloading processed_meme_data.pkl from Google Drive with ID: %s", file_id)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
    except Exception as e:
        st.error(f"Failed to download processed_meme_data.pkl (ID: {file_id}): {e}")
        logger.error("Failed to download processed_meme_data.pkl: %s", str(e))
        return None
    
    if os.path.exists(output_file):
        st.success(f"Downloaded processed_meme_data.pkl to {output_file}")
        logger.info("Downloaded processed_meme_data.pkl to %s", output_file)
        try:
            with open(output_file, 'rb') as f:
                processed_data = pickle.load(f)
            return processed_data
        except Exception as e:
            st.error(f"Failed to load processed data from pickle file: {e}")
            logger.error("Pickle load error: %s", str(e))
            return None
    else:
        st.error("Failed to download processed_meme_data.pkl from Google Drive. Check File ID and sharing permissions.")
        logger.error("Pickle file not found: %s", output_file)
        return None

## ⚙️ Data Loading and Initialization

# Load CSV data and trigger downloads
@st.cache_data
def load_data(image_file_id, processed_file_id):
    try:
        # NOTE: Assumes these CSV files are included locally in a 'data/' directory
        synthetic_df = pd.read_csv('data/synthetic_meme_dataset_with_images.csv')
        signals_df = pd.read_csv('data/tradeable_signals.csv')
        mapped_df = pd.read_csv('data/mapped_meme_data.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading CSV files: {e}. Ensure data files are in the 'data/' folder.")
        logger.error("Error loading CSV files: %s", str(e))
        return None, None, None, None
    
    # Trigger download of the processed data (Model/features)
    processed_data = download_processed_data(processed_file_id)
    
    return synthetic_df, signals_df, mapped_df, processed_data

# Execute downloads and load dataframes
image_folder = download_meme_images(MEME_IMAGE_FILE_ID)
if image_folder is None:
    st.warning("Could not download meme images. Displaying locally if available.")
    image_folder = "meme_images" # Fallback to local folder if download fails

synthetic_df, signals_df, mapped_df, processed_data = load_data(MEME_IMAGE_FILE_ID, PROCESSED_DATA_FILE_ID)

# Critical stop if essential data is missing
if synthetic_df is None or processed_data is None:
    st.error("Critical data failed to load. Please check file paths and Google Drive IDs/permissions.")
    st.stop() 

# --- APPLICATION SECTIONS ---

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
    elif isinstance(processed_data, dict):
        st.write("Processed data is a dictionary (likely containing model/features). Keys found:")
        st.code(list(processed_data.keys()))
    else:
        st.write(f"Processed data type: **{type(processed_data)}**")

---

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
        # Use log scale on y-axis for better visibility if data is heavily skewed
        sns.histplot(synthetic_df['price_change'], kde=True, ax=ax) 
        ax.set_title("Price Change Distribution")
        st.pyplot(fig)
    else:
        st.warning("Column 'price_change' not found in synthetic data for histogram.")

---

elif page == "Meme Gallery":
    st.header("🖼️ Meme Gallery")
    st.markdown("A selection of images used in the analysis.")
    
    if os.path.isdir(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        image_files = []
        
    if image_files:
        st.info(f"Loaded {len(image_files)} images from the `{image_folder}` folder.")
        
        # Display images in columns
        cols = st.columns(4)
        for i, filename in enumerate(image_files[:12]): # Show top 12 images
            try:
                img_path = os.path.join(image_folder, filename)
                image = Image.open(img_path)
                with cols[i % 4]:
                    st.image(image, caption=filename, use_column_width=True)
            except Exception as e:
                logger.error(f"Failed to display image {filename}: {e}")
                
    else:
        st.error(f"No images found in the '{image_folder}' folder. The download may have failed.")

---

elif page == "Trading Signals":
    st.header("🚦 Trading Signals")
    st.markdown("Signals generated from the processed data that suggest buy/sell opportunities.")
    
    if signals_df is not None:
        st.subheader(f"Latest {len(signals_df)} Trading Signals")
        st.dataframe(signals_df)
    else:
        st.error("Trading Signals data not loaded.")

---

elif page == "Model Insights":
    st.header("🧠 Model Insights")
    st.markdown("Feature importance and model performance metrics.")
    
    # Check if processed_data is a dictionary and contains a model
    if isinstance(processed_data, dict) and 'model' in processed_data and 'features' in processed_data:
        model = processed_data['model']
        
        # Feature Importance
        st.subheader("Feature Importance")
        try:
            # Assuming model is an XGBoost or similar object with feature_importances_
            feature_importances = pd.Series(model.feature_importances_, index=processed_data['features'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importances.sort_values(ascending=False).head(10).plot(kind='barh', ax=ax)
            ax.set_title("Top 10 Feature Importances")
            st.pyplot(fig)
            
        except AttributeError:
            st.warning("Model object does not have the 'feature_importances_' attribute.")
        except Exception as e:
            st.warning(f"Failed to display feature importance: {e}")
            
        # Model Metrics (Placeholder/Example)
        st.subheader("Model Metrics (Example)")
        st.markdown("""
        | Metric | Value |
        |---|---|
        | **Accuracy** | 85.5% |
        | **Precision** | 82.1% |
        | **Recall** | 88.0% |
        | **F1-Score** | 85.0% |
        """)
        
    else:
        st.warning("Model or required feature data ('model', 'features' keys) not found in the 'processed_data'.")
