import streamlit as st
import pandas as pd
import os
import pickle
import gdown
import zipfile
import logging
import plotly.express as px
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
# -------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & FILE IDs ---
# Set page config for a cleaner, wider layout
st.set_page_config(page_title="Meme Stock Insights", layout="wide", initial_sidebar_state="expanded")

# NOTE: Replace these with your actual, publicly shared Google Drive IDs.
MEME_IMAGE_FILE_ID = "17y_b9nmOBx_ethy6tfv_Big8teFiD2OR"
PROCESCESSED_DATA_FILE_ID = "1TBIKxWxPeF6e70Y0NybPVFjKaMW8pCz3"
# --- END FILE IDs ---

# --- Data Download Functions ---
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

# --- Data Loading and Initialization ---
@st.cache_data
def load_data(image_file_id, processed_file_id):
    try:
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


# --- APPLICATION HEADER & NAVIGATION ---
st.title("📈 Meme Stock Insights Dashboard")
st.markdown("""
Welcome to the modern insights platform! Explore datasets, interactive visualizations, and powerful model insights.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", [
    "Datasets", 
    "Visualizations", 
    "Meme Gallery", 
    "Upload & Explore", 
    "Model Insights"
])

# --- APPLICATION SECTIONS ---

if page == "Datasets":
    st.header("📂 Data Overview & Trading Signals")
    
    # Use tabs for clean organization of different dataframes
    tab1, tab2, tab3, tab4 = st.tabs(["Main Data", "Mapped Data", "Trading Signals", "Processed Data"])
    
    with tab1:
        st.subheader("Main Meme Stock Data")
        st.markdown("Raw data combining stock price movements and Reddit post features.")
        st.dataframe(main_data_df)
    
    with tab2:
        st.subheader("Mapped Meme Data")
        st.dataframe(mapped_df)
    
    with tab3:
        st.subheader("Tradeable Signals")
        st.markdown("Model-generated signals for potential buy/sell opportunities.")
        st.dataframe(signals_df)
    
    with tab4:
        st.subheader("Processed Data (from Pickle)")
        if isinstance(processed_data, pd.DataFrame):
            st.dataframe(processed_data)
        elif isinstance(processed_data, dict):
            st.write("Processed data is a dictionary (Model/Features). Keys found:")
            st.code(list(processed_data.keys()))
        else:
            st.write(f"Processed data type: **{type(processed_data)}**")
        
    st.markdown("---") 

# ----------------------------------------------------------------------
elif page == "Visualizations":
    st.header("📊 Interactive Visualizations")
    st.markdown("Explore key distributions and trends using interactive Plotly charts.")
    
    # ------------------ Row 1: Key Metrics ------------------
    st.subheader("Key Data Metrics")
    
    # Calculate key metrics (assuming 'price_change' is percentage or decimal)
    avg_change = main_data_df['price_change'].mean() * 100
    max_change = main_data_df['price_change'].max() * 100
    total_posts = len(main_data_df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Points", total_posts)
    col2.metric("Average Daily Change", f"{avg_change:.2f}%", delta=f"{avg_change:.2f}%")
    col3.metric("Max Daily Change Observed", f"{max_change:.2f}%", delta=f"+{max_change:.2f}%")

    st.markdown("---")
    
    # ------------------ Row 2: Charts ------------------
    col_vis1, col_vis2 = st.columns(2)
    
    # 1. Sentiment Distribution (Plotly Bar Chart)
    with col_vis1:
        st.subheader("1. Reddit Post Sentiment Distribution")
        if 'sentiment' in main_data_df.columns:
            sentiment_counts = main_data_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            fig = px.bar(
                sentiment_counts,
                x='Sentiment',
                y='Count',
                color='Sentiment',
                title="Sentiment Count",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'sentiment' not found in main data.")
    
    # 2. Price Change Histogram (Plotly Histogram)
    with col_vis2:
        st.subheader("2. Price Change Distribution")
        if 'price_change' in main_data_df.columns:
            # Create a histogram, clipping the data for better visualization
            df_hist = main_data_df['price_change'].clip(lower=-0.5, upper=0.5)
            
            fig_hist = px.histogram(
                df_hist, 
                x='price_change', 
                nbins=50, 
                title="Capped Daily Price Change Distribution (-50% to +50%)",
                template="plotly_white",
            )
            fig_hist.update_layout(xaxis_title="Daily Price Change (Decimal)")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Column 'price_change' not found in main data.")
            
    st.markdown("---")
    
    # ------------------ Row 3: Trend Line Plot ------------------
    st.subheader("3. Interactive Price Trend Analysis")
    
    if 'date' in main_data_df.columns and 'price_change' in main_data_df.columns and 'stock_ticker' in main_data_df.columns:
        
        # Select ticker for plot
        unique_tickers = main_data_df['stock_ticker'].unique()
        selected_ticker = st.selectbox("Select Stock Ticker to Analyze Trend", unique_tickers, index=0)
        
        df_plot = main_data_df[main_data_df['stock_ticker'] == selected_ticker].copy()
        df_plot['date'] = pd.to_datetime(df_plot['date'], errors='coerce')
        df_plot = df_plot.dropna(subset=['date']).sort_values(by='date')
        
        # Calculate 7-Day Rolling Average
        df_plot['7-Day Rolling Change'] = df_plot['price_change'].rolling(window=7, min_periods=1).mean()
        
        fig_line = px.line(
            df_plot,
            x='date',
            y='7-Day Rolling Change',
            title=f"7-Day Rolling Avg. Price Change for {selected_ticker}",
            template="plotly_white",
        )
        fig_line.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Change")
        fig_line.update_layout(yaxis_title="Rolling Avg. Change")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Required columns 'date', 'price_change', or 'stock_ticker' not found for Trend Plot.")


# ----------------------------------------------------------------------

elif page == "Meme Gallery":
    st.header("🖼️ Meme Gallery")
    st.markdown("A selection of images used in the analysis.")
    
    if os.path.isdir(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        image_files = []
        
    if image_files:
        st.info(f"Loaded {len(image_files)} images from the `{image_folder}` folder.")
        
        # Use st.expander for a cleaner display
        with st.expander("Click to view sample memes", expanded=True):
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

# ----------------------------------------------------------------------
elif page == "Upload & Explore":
    st.header("⬆️ Upload & Explore Custom Data")
    st.markdown("Upload a new CSV file to view its details and basic statistics.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv"
    )
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded '{uploaded_file.name}' with {len(uploaded_df)} rows and {len(uploaded_df.columns)} columns.")
            
            st.markdown("---")
            
            # Use tabs for organized display
            up_tab1, up_tab2, up_tab3 = st.tabs(["Data Preview", "Data Types & Nulls", "Descriptive Stats"])
            
            with up_tab1:
                st.subheader("Uploaded Data Preview")
                st.dataframe(uploaded_df.head(10))
            
            with up_tab2:
                st.subheader("Column Information")
                # Prepare dtypes and nulls in one table
                info_df = uploaded_df.dtypes.astype(str).to_frame(name='Data Type')
                info_df['Non-Null Count'] = uploaded_df.count()
                info_df['Null Count'] = uploaded_df.isnull().sum()
                st.dataframe(info_df)
                
            with up_tab3:
                st.subheader("Descriptive Statistics (Numeric)")
                st.dataframe(uploaded_df.describe().T)

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

    st.markdown("---") 

# ----------------------------------------------------------------------

elif page == "Model Insights":
    st.header("🧠 Model Performance & Feature Importance")
    st.markdown("Detailed breakdown of the predictive model's performance on the test set.")
    
    required_keys = ['model', 'features', 'y_test', 'y_pred', 'y_proba']
    if (isinstance(processed_data, dict) and 
        all(key in processed_data for key in required_keys)):
        
        model = processed_data['model']
        y_test = processed_data['y_test']
        y_pred = processed_data['y_pred']
        y_proba = processed_data['y_proba']
        
        # --- CRITICAL FIX FOR INDEXERROR ---
        # Checks the shape of y_proba to handle both 1-column and 2-column formats
        try:
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                # Standard format: [:, 0] is proba for class 0, [:, 1] is proba for class 1
                y_proba_positive = y_proba[:, 1] 
            elif len(y_proba.shape) == 1 or (len(y_proba.shape) == 2 and y_proba.shape[1] == 1):
                # Saved as 1-column array (already positive probability)
                y_proba_positive = y_proba.flatten() 
            else:
                st.error("Model probabilities are in an unexpected format. Cannot calculate ROC AUC.")
                st.stop()
        except AttributeError:
             st.error("Model probabilities ('y_proba') is not a valid numpy array. Cannot calculate ROC AUC.")
             st.stop()
        # -----------------------------------
        
        # Calculate scores
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_proba_positive) # Use the corrected array
        roc_auc = auc(fpr, tpr)
        
        # --- Row 1: Key Performance Metrics ---
        st.subheader("Key Classification Metrics")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

        col_m1.metric("Accuracy", f"{accuracy:.3f}")
        col_m2.metric("Precision", f"{precision:.3f}")
        col_m3.metric("Recall", f"{recall:.3f}")
        col_m4.metric("F1-Score", f"{f1:.3f}")
        col_m5.metric("ROC AUC", f"{roc_auc:.3f}")

        st.markdown("---")

        # --- Row 2: Charts (Feature Importance & ROC) ---
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "ROC Curve"])

        with chart_tab1:
            st.subheader("Top 10 Feature Importances")
            try:
                feature_importances = pd.Series(model.feature_importances_, index=processed_data['features']).sort_values(ascending=False).head(10)
                feature_df = feature_importances.reset_index()
                feature_df.columns = ['Feature', 'Importance']

                # Plotly Bar Chart for Feature Importance
                fig_feat = px.bar(
                    feature_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Predictive Features",
                    template="plotly_white"
                )
                fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_feat, use_container_width=True)
            except Exception as e:
                st.warning(f"Failed to display feature importance: {e}")

        with chart_tab2:
            st.subheader("Confusion Matrix")
            try:
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, 
                                     index=['True Down (0)', 'True Up (1)'], 
                                     columns=['Pred Down (0)', 'Pred Up (1)'])
                
                # Use Plotly Heatmap for better visual
                fig_cm = px.imshow(cm,
                                   text_auto=True, 
                                   labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                   x=['Down (0)', 'Up (1)'],
                                   y=['Down (0)', 'Up (1)'],
                                   color_continuous_scale='Blues')
                fig_cm.update_layout(title='Confusion Matrix')
                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.markdown("**Matrix Data:**")
                st.dataframe(cm_df)

            except Exception as e:
                st.error(f"Failed to display Confusion Matrix: {e}")
                
        with chart_tab3:
            st.subheader("ROC Curve")
            try:
                # Plotly ROC Curve
                fig_roc = px.area(
                    x=fpr, 
                    y=tpr, 
                    title=f'ROC Curve (AUC={roc_auc:.3f})', 
                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                    width=600, height=500
                )
                fig_roc.add_shape(
                    type='line', line=dict(dash='dash'), 
                    x0=0, x1=1, y0=0, y1=1
                )
                fig_roc.update_layout(template="plotly_white")
                st.plotly_chart(fig_roc)

            except Exception as e:
                st.error(f"Failed to display ROC Curve: {e}")

    else:
        st.warning("""
        Model or required test data not found in the 'processed_data'. 
        Please ensure your Jupyter Notebook saves the following keys to 'processed_meme_data.pkl':
        - **'model'** (Trained Model Object)
        - **'features'** (List of feature names)
        - **'y_test'** (True labels of the test set)
        - **'y_pred'** (Predicted labels of the test set)
        - **'y_proba'** (Prediction probabilities - either 1 or 2 columns)
        
        **Action Required:** If you see this warning, please ensure your model training notebook is saved correctly and the file is accessible via the Google Drive ID.
        """)
