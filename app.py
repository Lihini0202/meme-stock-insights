import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Kept for potential Matplotlib/Seaborn use, but Plotly is preferred
import seaborn as sns          # Kept for potential Matplotlib/Seaborn use, but Plotly is preferred
from PIL import Image
import os
import pickle
import gdown
import zipfile
import xgboost as xgb
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# --- NEW MODERN CHART IMPORTS ---
import plotly.express as px
import plotly.graph_objects as go
# -------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEME_IMAGE_FILE_ID = "17y_b9nmOBx_ethy6tfv_Big8teFiD2OR"
PROCESSED_DATA_FILE_ID = "1TBIKxWxPeF6e70Y0NybPVFjKaMW8pCz3"

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
page = st.sidebar.radio(
    "Select Section", 
    [
        "📂 Datasets", 
        "📉 Visualizations", 
        "🖼️ Meme Gallery", 
        "🚦 Trading Signals", 
        "🧠 Model Insights"
    ]
)

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

if page == "📂 Datasets":
    st.header("📂 Datasets Overview")
    
    with st.expander("Main Meme Stock Data (Head)"):
        st.write("Current Columns in main_data_df:")
        st.code(main_data_df.columns.tolist())
        st.dataframe(main_data_df.head(10))
    
    with st.expander("Mapped Meme Data (Head)"):
        st.dataframe(mapped_df.head(10))
    
    with st.expander("Tradeable Signals (Head)"):
        st.dataframe(signals_df.head(10))
    
    with st.expander("Processed Meme Data (from Pickle)"):
        if isinstance(processed_data, pd.DataFrame):
            st.dataframe(processed_data.head(10))
        elif isinstance(processed_data, dict):
            st.write("Processed data is a dictionary (likely containing model/features). Keys found:")
            st.code(list(processed_data.keys()))
        else:
            st.write(f"Processed data type: **{type(processed_data)}**")
            
    st.markdown("---") 


elif page == "📉 Visualizations":
    st.header("📉 Interactive Visualizations")
    st.markdown("Exploring key distributions and trends using **Plotly** for interactivity.")
     
    # Row 1: Sentiment Distribution and Price Change Histogram (PLotly)
    col1, col2 = st.columns(2)
     
    # 1. Sentiment Distribution (Plotly Histogram)
    with col1:
        st.subheader("1. Sentiment Distribution")
        if 'sentiment' in main_data_df.columns:
            try:
                fig = px.histogram(
                    main_data_df, 
                    x='sentiment', 
                    title="Reddit Post Sentiment Distribution",
                    color='sentiment', # Color bars based on sentiment value
                    color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}
                )
                fig.update_layout(xaxis={'categoryorder':'total descending'}, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating Sentiment Distribution: {e}")
        else:
            st.warning("Column 'sentiment' not found in main data.")
     
    # 2. Price Change Histogram (Plotly Histogram)
    with col2:
        st.subheader("2. Price Change Histogram")
        if 'price_change' in main_data_df.columns:
            try:
                # Limit range for better visibility and avoid extreme outliers
                plot_data = main_data_df['price_change'].clip(lower=-0.5, upper=0.5).to_frame(name='Capped Price Change')
                fig = px.histogram(
                    plot_data, 
                    x='Capped Price Change', 
                    title="Capped Daily Price Change Distribution (-50% to +50%)",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating Price Change Histogram: {e}")
        else:
            st.warning("Column 'price_change' not found in main data.")

    st.markdown("---") 

    # Row 2: Average Change by Stock and Price Trend Sample (Plotly)
    col3, col4 = st.columns(2)

    # 3. Average Price Change by Stock (Plotly Bar Plot)
    with col3:
        st.subheader("3. Average Daily Change by Stock")
        if 'stock_ticker' in main_data_df.columns and 'price_change' in main_data_df.columns:
            try:
                # Group and calculate mean, then limit to top 10 for cleaner display
                avg_change = main_data_df.groupby('stock_ticker')['price_change'].mean().sort_values(ascending=False).head(10).reset_index()
                avg_change.columns = ['Stock Ticker', 'Avg. Price Change']

                fig = px.bar(
                    avg_change, 
                    x='Stock Ticker', 
                    y='Avg. Price Change', 
                    title="Top 10 Avg. Price Change by Ticker",
                    color='Avg. Price Change',
                    color_continuous_scale=px.colors.diverging.RdYlGn # Good for positive/negative change
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating Average Change by Stock: {e}")
        else:
            st.warning("Required columns 'stock_ticker' or 'price_change' not found for Bar Plot.")

    # 4. Price Trend Over Time (Plotly Line Plot)
    with col4:
        st.subheader("4. Price Trend Sample")
        date_col = 'date' 
         
        if date_col in main_data_df.columns and 'price_change' in main_data_df.columns and 'stock_ticker' in main_data_df.columns:
            try:
                df_plot = main_data_df.copy()
                df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
                df_plot = df_plot.dropna(subset=[date_col])
                df_plot = df_plot.sort_values(by=['stock_ticker', date_col])
                df_plot = df_plot.set_index(date_col)

                # Plot the rolling average of price change for a sample stock
                sample_ticker = df_plot['stock_ticker'].iloc[0] # Get the first ticker
                trend_data = df_plot[df_plot['stock_ticker'] == sample_ticker]['price_change'].rolling(window=7).mean().dropna().reset_index()
                
                fig = px.line(
                    trend_data, 
                    x=date_col, 
                    y='price_change', 
                    title=f"7-Day Rolling Avg. Price Change for {sample_ticker}"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Change")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating Price Trend Sample: {e}")
        else:
            st.warning("Required columns 'date', 'price_change', or 'stock_ticker' not found for Trend Plot.")

    st.markdown("---") 



elif page == "🖼️ Meme Gallery":
    st.header("🖼️ Meme Gallery")
    st.markdown("A selection of images used in the analysis.")
     
    if os.path.isdir(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        image_files = []
         
    if image_files:
        st.info(f"Loaded **{len(image_files)}** images from the `{image_folder}` folder.")
         
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



elif page == "🚦 Trading Signals":
    st.header("🚦 Trading Signals")
    st.markdown("Signals generated from the processed data that suggest buy/sell opportunities.")
     
    if signals_df is not None:
        st.subheader(f"Latest {len(signals_df)} Trading Signals")
        st.dataframe(signals_df) # Streamlit dataframe is modern enough here
    else:
        st.error("Trading Signals data not loaded.")

    st.markdown("---") 



elif page == "🧠 Model Insights":
    st.header("🧠 Model Insights")
    st.markdown("Feature importance and performance metrics for the stock prediction model.")
     
    required_keys = ['model', 'features', 'y_test', 'y_pred', 'y_proba']
    if (isinstance(processed_data, dict) and 
        all(key in processed_data for key in required_keys)):
         
        model = processed_data['model']
        y_test = processed_data['y_test']
        y_pred = processed_data['y_pred']
        y_proba = processed_data['y_proba']
         
        # Calculate scores
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # --- ROW 1: Key Performance Indicators (KPIs) ---
        st.subheader("Key Model Performance Metrics")
        
        col_acc, col_prec, col_rec, col_f1, col_auc = st.columns(5)

        col_acc.metric("Accuracy", f"{accuracy:.3f}")
        col_prec.metric("Precision", f"{precision:.3f}", help="Ability of the model to avoid false positives (signals that didn't pan out).")
        col_rec.metric("Recall", f"{recall:.3f}", help="Ability of the model to find all positive samples (finding all true stock increases).")
        col_f1.metric("F1-Score", f"{f1:.3f}")
        col_auc.metric("ROC AUC", f"{roc_auc:.3f}", help="Measure of separability between classes.")
        
        st.markdown("---")

        # --- ROW 2: Feature Importance (Modernized via Plotly) ---
        st.subheader("Feature Importance")
        try:
            feature_importances = pd.Series(model.feature_importances_, index=processed_data['features']).sort_values(ascending=False).head(10)
            importance_df = feature_importances.reset_index()
            importance_df.columns = ['Feature', 'Importance']
            
            fig_feat = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h', 
                title="Top 10 Feature Importances",
                color='Importance'
            )
            fig_feat.update_layout(yaxis={'categoryorder':'total ascending'}) # Display largest bar at top
            st.plotly_chart(fig_feat, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Failed to display feature importance: {e}")

        st.markdown("---")

        # --- ROW 3: Model Performance Visualizations (Confusion Matrix and ROC) ---
        col1, col2 = st.columns(2)

        # 1. Confusion Matrix (Plotly Heatmap)
        with col1:
            st.subheader("Confusion Matrix")
            try:
                cm_labels = ['Down (0)', 'Up (1)']
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True, 
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=cm_labels,
                    y=cm_labels,
                    color_continuous_scale='Blues'
                )
                fig_cm.update_xaxes(side="bottom")
                fig_cm.update_layout(title_text='Confusion Matrix')
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to display Confusion Matrix: {e}")
                 
        # 2. ROC Curve (Plotly Scatter/Line)
        with col2:
            st.subheader("ROC Curve")
            try:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
                fig_roc.update_layout(
                    title='Receiver Operating Characteristic', 
                    xaxis_title='False Positive Rate', 
                    yaxis_title='True Positive Rate',
                    legend=dict(x=1, y=0, orientation='v')
                )
                st.plotly_chart(fig_roc, use_container_width=True)
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
        - **'y_proba'** (Prediction probabilities for the positive class)
        """)


