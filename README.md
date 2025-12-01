# 🎉 Meme Stock Insights: Modern Dashboard 🚀💎🙌

*Streamlit app for exploring meme stock data with interactive visualizations and in-depth model performance analysis.*

---

## 🌟 Overview & Purpose ✨

**Meme Stock Insights** is a powerful, refactored Streamlit application designed for in-depth analysis of meme stock data, leveraging data preparation and model training performed in Google Colab.

The app provides a **wonderful, modern, multi-tab experience** to explore datasets, visualize key trends using **interactive Plotly charts**, browse associated **memes**, and gain **critical model performance insights**.

---

## 🚀 Key Features of the Modern App 📊

| Section | Feature Highlight | Description |
| :--- | :--- | :--- |
| 📂 **Datasets** | **Organized Data Views** | Access raw data, mapped features, trading signals, and the pickled model contents via a clean tab interface (`st.tabs`). |
| 📊 **Visualizations** | **Interactive Plotly Charts** | Explore data distributions (sentiment, price change) and time series trends using dynamic, zoomable charts for superior user experience. |
| 🧠 **Model Insights** | **Comprehensive Performance** | View key metrics (Accuracy, F1, ROC AUC), **Top Feature Importance**, and a Plotly-rendered **Confusion Matrix** and **ROC Curve** for classification evaluation. |
| 🖼️ **Meme Gallery** | **Associated Imagery** | Browse stock-related images used in the analysis via a clean, collapsible gallery, powered by Streamlit's `st.expander`. |
| ⬆️ **Upload & Explore** | **Custom Data Upload** | Upload your own CSV files for quick inspection of descriptive statistics and column information. |

---

## 💡 Technical Deep Dive 🧠

This application employs several modern Streamlit and Python techniques to ensure a seamless, high-performance experience:

* **Interactive Plotting:** Instead of static `matplotlib` images, we use the **Plotly Express** library (`plotly.express as px`) to generate all visualizations. This provides users with built-in **zoom, pan, hover, and data-point visibility toggles** instantly, enhancing data exploration.
* **Robust Data Loading:** We use the `@st.cache_resource` decorator to efficiently download large `.pkl` (model data) and `.zip` (image assets) files from Google Drive using the `gdown` library. This ensures the files are only downloaded once, drastically improving load times on subsequent runs.
* **Model Probability Handling (The Fix):** The "Model Insights" section includes crucial logic to robustly handle the `y_proba` array saved from the model. It automatically checks if the array is saved as a **one-column** (positive class probability only) or **two-column** (all class probabilities) format to prevent the common `IndexError` during the calculation of the **ROC Curve and AUC**.
* **Modern UI Components:** The application utilizes native Streamlit components like `st.set_page_config(layout="wide")`, `st.tabs`, and `st.metric` to create a clean, modern dashboard aesthetic with organized, segmented content.

## 📂 Folder Structure

```
meme-stock-insights/
├── app.py                      # 🌟 Main Streamlit dashboard
├── requirements.txt            # 📦 Dependencies (plotly, gdown, scikit-learn, etc.)
├── data/                       # 💾 All required dataset files
│   ├── mapped_meme_data.csv                # Cleaned & mapped meme data
│   ├── processed_meme_data.pkl             # Pickled ML components (model, y_test, y_proba…)
│   ├── meme_dataset_with_images.csv        # Dataset linking memes to image URLs/files
│   └── tradeable_signals.csv               # Generated trading signals
├── meme_images/                # 🎨 Folder for downloaded meme images
│   └── (e.g., GME_meme.png)
└── README.md                   # 📖 Project guide
```


## 🛠️ Setup & Local Installation ⚡

### 🎯 Prerequisites
- 🐍 **Python 3.11** or later
- **Data Files:** Ensure your Google Drive IDs for the image zip and `processed_meme_data.pkl` are correct within `app.py` and publicly accessible.

### 💻 Installation Steps

1. Clone the repository:
   ```bash
   git clone [https://github.com/Lihini0202/meme-stock-insights.git](https://github.com/Lihini0202/meme-stock-insights.git)
   cd meme-stock-insights
   ```
2. Install all required dependencies (including Plotly and scikit-learn):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment on Streamlit Cloud 🌍

Deploying **Meme Stock Insights** is simple and fast:

1. **Commit & Push Your Code**
   Make sure all updates — especially `app.py` and `requirements.txt` — are pushed to your GitHub repository.

2. **Connect Your Repo on Streamlit Cloud**
   Go to **[https://share.streamlit.io](https://share.streamlit.io)**, log in, and select your GitHub repository for deployment.

3. **Automatic Environment Setup**
   Streamlit Cloud will automatically:

   * Install dependencies from `requirements.txt`
   * Download meme images and dataset files using Google Drive File IDs in `app.py`
   * Build and launch the dashboard

4. **Deploy & Share**
   Once the build completes, Streamlit provides a **public shareable link** to your live dashboard.

---

## 🙌 Acknowledgments

Built with:

* ❤️ **Streamlit** for building a clean, modern dashboard UI
* 📊 **Plotly** for interactive, high-quality visualizations
* 🤖 **Google Colab + scikit-learn** for model training and data preparation
* 🚀 Inspired by the exciting, unpredictable world of **meme stocks**
