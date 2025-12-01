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
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
# -------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GOOGLE DRIVE FILE IDs --- 
MEME_IMAGE_FILE_ID = "17y_b9nmOBx_ethy6tfv_Big8teFiD2OR" 
PROCESSED_DATA_FILE_ID = "1TBIKxWxPeF6e70Y0NybPVFjKaMW8pCz3"
# --- END FILE IDs ---

# Set page config
st.set_page_config(page_title="Meme Stock Insights", layout="wide")

# Title and description
st.title("📈 Meme Stock Insights Dashboard")
st.markdown("""
This app provides insights into meme stocks based on real-world and processed data. 
Explore datasets, visualizations, trading signals, and model insights.
Meme images and processed data are downloaded from Google Drive due to size constraints.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", ["Datasets", "Visualizations", "Meme Gallery", "Trading Signals", "Model Insights"])

## 📂 Data Download Functions

@st.cache_resource
def download_meme_images(file_id):
    output_zip = "meme_image.zip"
    output_folder = "meme_images"
    
    try:
        logger.info("Downloading meme_images.zip from Google Drive with ID: %s", file_id)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip, quiet=False)
    except Exception as e:
        st.error(f"Failed to download meme_images.zip (ID: {file_id}): {e}")
        logger.error("Failed to download meme_images.zip: %s", str(e))
        return None
    
    if os.path.exists(output_zip):
        try:
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                os.makedirs(output_folder, exist_ok=True)
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

@st.cache_resource
def download_processed_data(file_id):
    output_file = "data/processed_meme_data.pkl"
    os.makedirs("data", exist_ok=True)
    
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

@st.cache_data
def load_data(image_file_id, processed_file_id):
    try:
        # Load the CSV files
        main_data_df = pd.read_csv('data/meme_dataset_with_images.csv')
        signals_df = pd.read_csv('data/tradeable_signals.csv')
        mapped_df = pd.read_csv('data/mapped_meme_data.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading CSV files: {e}. Ensure data files are in the 'data/' folder.")
        logger.error("Error loading CSV files: %s", str(e))
        return None, None, None, None
    

    try:
        main_data_df.rename(columns={
            'ticker': 'stock_ticker',       
            'timestamp': 'date'             
          
        }, inplace=True)
    except KeyError as e:
        st.warning(f"Failed to find and rename a required column: {e}. Please ensure 'ticker' and 'timestamp' are in your CSV.")
        
    # Processed data must be downloaded/loaded
    processed_data = download_processed_data(processed_file_id)
    
    return main_data_df, signals_df, mapped_df, processed_data

# Execute downloads and load dataframes
image_folder = download_meme_images(MEME_IMAGE_FILE_ID)
if image_folder is None:
    st.warning("Could not download meme images. Displaying locally if available.")
    image_folder = "meme_images"

os.makedirs("data", exist_ok=True)
main_data_df, signals_df, mapped_df, processed_data = load_data(MEME_IMAGE_FILE_ID, PROCESSED_DATA_FILE_ID)

if main_data_df is None or processed_data is None:
    st.error("Critical data failed to load. Please check file paths and Google Drive IDs/permissions.")
    st.stop() 

# --- APPLICATION SECTIONS ---

if page == "Datasets":
    st.header("📊 Datasets Overview")
    
    st.subheader("Main Meme Stock Data")
    st.write("Current Columns in main_data_df:")
    st.code(main_data_df.columns.tolist())
    st.dataframe(main_data_df.head(10))
    
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
        
    st.markdown("---") 


elif page == "Visualizations":
    st.header("📉 Visualizations")
    st.markdown("Exploring key distributions and trends in the dataset.")
    
    # Row 1: Sentiment Distribution and Price Change Histogram
    col1, col2 = st.columns(2)
    
    # 1. Sentiment Distribution (Count Plot)
    with col1:
        st.subheader("1. Sentiment Distribution")
        if 'sentiment' in main_data_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(6, 4)) # Smaller figure size
                sns.countplot(x='sentiment', data=main_data_df, ax=ax, palette='viridis')
                ax.set_title("Reddit Post Sentiment Distribution")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating Sentiment Distribution: {e}")
        else:
            st.warning("Column 'sentiment' not found in main data.")
    
    # 2. Price Change Histogram
    with col2:
        st.subheader("2. Price Change Histogram")
        if 'price_change' in main_data_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(6, 4)) # Smaller figure size
                # Limit range for better visibility and avoid extreme outliers
                sns.histplot(main_data_df['price_change'].clip(lower=-0.5, upper=0.5), kde=True, ax=ax, bins=30) 
                ax.set_title("Capped Price Change Distribution (-50% to +50%)")
                ax.set_xlabel("Daily Price Change")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating Price Change Histogram: {e}")
        else:
            st.warning("Column 'price_change' not found in main data.")

    st.markdown("---") 

    # Row 2: Average Change by Stock and Price Trend Sample
    col3, col4 = st.columns(2)

    # 3. Average Price Change by Stock (Bar Plot)
    with col3:
        st.subheader("3. Average Daily Change by Stock")
        if 'stock_ticker' in main_data_df.columns and 'price_change' in main_data_df.columns:
            try:
                # Group and calculate mean, then limit to top 10 for cleaner display
                avg_change = main_data_df.groupby('stock_ticker')['price_change'].mean().sort_values(ascending=False).head(10)
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4)) # Smaller figure size
                sns.barplot(x=avg_change.index, y=avg_change.values, ax=ax_bar, palette='coolwarm')
                ax_bar.set_title("Top 10 Avg. Price Change by Ticker")
                ax_bar.set_xlabel("Stock Ticker")
                ax_bar.set_ylabel("Avg. Price Change")
                ax_bar.tick_params(axis='x', rotation=45)
                st.pyplot(fig_bar)
            except Exception as e:
                st.error(f"Error generating Average Change by Stock: {e}")
        else:
            st.warning("Required columns 'stock_ticker' or 'price_change' not found for Bar Plot.")

    # 4. Price Trend Over Time (Line Plot)
    with col4:
        st.subheader("4. Price Trend Sample")
        date_col = 'date' 
        
        if date_col in main_data_df.columns and 'price_change' in main_data_df.columns and 'stock_ticker' in main_data_df.columns:
            try:
                df_plot = main_data_df.copy()
                df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
                df_plot = df_plot.dropna(subset=[date_col])
                
                
                df_plot = df_plot.sort_values(by=['stock_ticker', date_col])
                # ---------------------------------------------------------------------------------
                
                # Set index for rolling calculation/plotting
                df_plot = df_plot.set_index(date_col)

                # Plot the rolling average of price change for a sample stock
                sample_ticker = df_plot['stock_ticker'].iloc[0] # Get the first ticker
                
                trend_data = df_plot[df_plot['stock_ticker'] == sample_ticker]['price_change'].rolling(window=7).mean().dropna()
                
                fig_line, ax_line = plt.subplots(figsize=(6, 4)) # Smaller figure size
                
                # Plotting against the index (the sorted dates)
                ax_line.plot(trend_data.index, trend_data.values, label=f'7-Day Rolling Avg. Price Change ({sample_ticker})')
                
                ax_line.axhline(0, color='red', linestyle='--', linewidth=0.5)
                ax_line.set_title(f"Price Change Trend for {sample_ticker}")
                ax_line.set_xlabel("Date")
                ax_line.set_ylabel("Rolling Avg. Change")
                fig_line.autofmt_xdate()
                ax_line.legend()
                st.pyplot(fig_line)

            except Exception as e:
                st.error(f"Error generating Price Trend Sample: {e}")
        else:
            st.warning("Required columns 'date', 'price_change', or 'stock_ticker' not found for Trend Plot.")

    st.markdown("---") 



elif page == "Meme Gallery":
    st.header("🖼️ Meme Gallery")
    st.markdown("A selection of images used in the analysis.")
    
    if os.path.isdir(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        image_files = []
        
    if image_files:
        st.info(f"Loaded {len(image_files)} images from the `{image_folder}` folder.")
        
        cols = st.columns(4)
        for i, filename in enumerate(image_files[:12]):
            try:
                img_path = os.path.join(image_folder, filename)
                image = Image.open(img_path)
                with cols[i % 4]:
                    st.image(image, caption=filename, use_column_width=True)
            except Exception as e:
                logger.error(f"Failed to display image {filename}: {e}")
                
    else:
        st.error(f"No images found in the '{image_folder}' folder. The download may have failed.")

    st.markdown("---") 



elif page == "Trading Signals":
    st.header("🚦 Trading Signals")
    st.markdown("Signals generated from the processed data that suggest buy/sell opportunities.")
    
    if signals_df is not None:
        st.subheader(f"Latest {len(signals_df)} Trading Signals")
        st.dataframe(signals_df)
    else:
        st.error("Trading Signals data not loaded.")

    st.markdown("---") 



elif page == "Model Insights":
    st.header("🧠 Model Insights")
    st.markdown("Feature importance and model performance metrics.")
    
    required_keys = ['model', 'features', 'y_test', 'y_pred', 'y_proba']
    if (isinstance(processed_data, dict) and 
        all(key in processed_data for key in required_keys)):
        
        model = processed_data['model']
        y_test = processed_data['y_test']
        y_pred = processed_data['y_pred']
        y_proba = processed_data['y_proba']
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # --- ROW 1: Feature Importance ---
        st.subheader("Feature Importance")
        try:
            feature_importances = pd.Series(model.feature_importances_, index=processed_data['features'])
            
            fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
            feature_importances.sort_values(ascending=False).head(10).plot(kind='barh', ax=ax_feat)
            ax_feat.set_title("Top 10 Feature Importances")
            st.pyplot(fig_feat)
            
        except Exception as e:
            st.warning(f"Failed to display feature importance: {e}")

        st.markdown("---")

        # --- ROW 2: Model Performance Metrics (Confusion Matrix and ROC) ---
        col1, col2 = st.columns(2)

        # 1. Confusion Matrix
        with col1:
            st.subheader("Confusion Matrix")
            try:
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Down (0)', 'Up (1)'], 
                            yticklabels=['Down (0)', 'Up (1)'], ax=ax_cm)
                ax_cm.set_title('Confusion Matrix')
                ax_cm.set_xlabel('Predicted Label')
                ax_cm.set_ylabel('True Label')
                st.pyplot(fig_cm)
            except Exception as e:
                st.error(f"Failed to display Confusion Matrix: {e}")
                
        # 2. ROC Curve
        with col2:
            st.subheader("ROC Curve")
            try:
                
                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_title('Receiver Operating Characteristic')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            except Exception as e:
                st.error(f"Failed to display ROC Curve: {e}")

        st.markdown("---")

        # --- ROW 3: Metric Scores Table ---
        st.subheader("Quantitative Model Metrics")
        
        # Calculate scores
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
            'Value': [
                f"{accuracy:.3f}", 
                f"{precision:.3f}", 
                f"{recall:.3f}", 
                f"{f1:.3f}", 
                f"{roc_auc:.3f}"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)


    else:
        st.warning("""
        Model or required test data not found in the 'processed_data'. 
        Please ensure your Jupyter Notebook saves the following keys to 'processed_meme_data.pkl':
        - **'model'** (Trained Model Object)
        - **'features'** (List of feature names)
        - **'y_test'** (True labels of the test set)
        - **'y_pred'** (Predicted labels of the test set)
        - **'y_proba'** (Prediction probabilities for the positive class)
        """)



