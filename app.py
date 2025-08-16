import streamlit as st
import pandas as pd
import pickle
import requests
import os
import matplotlib.pyplot as plt
from PIL import Image
import zipfile

# Streamlit app title
st.title("Meme Data Explorer with Trained Files")

# Load trained files dynamically from Google Drive
@st.cache_data
def load_data():
    # Download processed_meme_data.pkl
    processed_url = "https://drive.google.com/uc?export=download&id=1DSpPKfF_CAzpOaPTRATkC9YjWR_bFND-"  # Confirmed ID
    processed_path = "data/processed_meme_data.pkl"
    if not os.path.exists(processed_path):
        response = requests.get(processed_url)
        with open(processed_path, 'wb') as f:
            f.write(response.content)

    # Download meme_images.zip (containing 1001 images)
    images_url = "https://drive.google.com/uc?export=download&id=1k5pODJE8e3omLmJotJZFfXrCZcM4XzoO"  # Replace with your meme_images.zip file ID  1k5pODJE8e3omLmJotJZFfXrCZcM4XzoO?usp=drive_link
    images_zip = "meme_img.zip"
    if not os.path.exists("meme_img"):
        response = requests.get(images_url)
        with open(images_zip, 'wb') as f:
            f.write(response.content)
        try:
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                # Test the zip file integrity
                if zip_ref.testzip() is not None:
                    st.error("The downloaded meme_images.zip file is corrupted. Please re-upload a valid zip file to Google Drive.")
                    return None, None, None, None
                zip_ref.extractall("meme_images/")
            os.remove(images_zip)
        except zipfile.BadZipFile:
            st.error("The file meme_images.zip is not a valid zip file or is corrupted. Please check the file on Google Drive.")
            return None, None, None, None
        except Exception as e:
            st.error(f"Error processing meme_images.zip: {str(e)}")
            return None, None, None, None

    # Load local files
    mapped_data = pd.read_csv("data/mapped_meme_data.csv")
    synthetic_data = pd.read_csv("data/synthetic_meme_dataset_with_images.csv") if os.path.exists("data/synthetic_meme_dataset_with_images.csv") else None
    signals_data = pd.read_csv("data/tradeable_signals.csv") if os.path.exists("data/tradeable_signals.csv") else None
    with open(processed_path, "rb") as f:
        processed_data = pickle.load(f)
    return mapped_data, processed_data, synthetic_data, signals_data

mapped_df, processed_df, synthetic_df, signals_df = load_data()

# Check if data loaded successfully
if mapped_df is None:
    st.error("Failed to load data. Please check the logs and Google Drive file IDs.")
else:
    # Display mapped dataset
    st.write("### Mapped Meme Data")
    st.dataframe(mapped_df)

    # Display processed dataset
    st.write("### Processed Meme Data")
    if isinstance(processed_df, pd.DataFrame):
        st.dataframe(processed_df)
    else:
        st.write("Processed data is not a DataFrame. Displaying as text:")
        st.write(processed_df)

    # Filtering
    st.write("### Filter Mapped Data")
    ticker_filter = st.multiselect("Select Tickers", options=mapped_df['ticker'].unique() if 'ticker' in mapped_df.columns else [], default=mapped_df['ticker'].unique() if 'ticker' in mapped_df.columns else [])
    sentiment_filter = st.multiselect("Select Sentiments", options=mapped_df['sentiment'].unique() if 'sentiment' in mapped_df.columns else [], default=mapped_df['sentiment'].unique() if 'sentiment' in mapped_df.columns else [])
    filtered_df = mapped_df[mapped_df['ticker'].isin(ticker_filter) & mapped_df['sentiment'].isin(sentiment_filter)] if 'ticker' in mapped_df.columns and 'sentiment' in mapped_df.columns else mapped_df
    st.dataframe(filtered_df)

    # Download filtered dataset
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Mapped Data",
        data=csv,
        file_name="filtered_mapped_meme_data.csv",
        mime="text/csv"
    )

    # Handle image-related data
    image_folder = "meme_images"
    os.makedirs(image_folder, exist_ok=True)
    if 'image_path' in mapped_df.columns:
        sample_image_path = os.path.join(image_folder, mapped_df.iloc[0]['image_path'])
        if os.path.exists(sample_image_path):
            st.image(sample_image_path, caption="Sample Meme Image")
        else:
            st.warning("Image path not found or inaccessible.")
    else:
        st.info("No image data available in mapped dataset.")

    # Visualization
    st.write("### Sentiment Distribution")
    if 'sentiment' in mapped_df.columns:
        fig, ax = plt.subplots()
        mapped_df['sentiment'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336', '#2196F3'])
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("No sentiment column available for visualization.")

    # Tradeable signals
    if signals_df is not None:
        st.write("### Tradeable Signals")
        st.dataframe(signals_df)
    else:
        st.info("Tradeable signals file not found.")



