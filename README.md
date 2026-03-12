# Customer Segmentation System

This project implements a full-stack data science solution for customer segmentation using RFM analysis and K-Means clustering.

## Features
- Synthetic Data Generation
- CSV Upload (with clear/reset support)
- RFM Analysis
- K-Means Clustering
- Interactive Streamlit Dashboard
- Business Insights

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`

### Notes
- Use **Clear uploaded data** before selecting a different file (or simply choose a new file from the uploader; the app will reset state automatically).
- Mapping fields correctly is required to avoid calculation errors; the app now clears previous mappings when a new dataset is loaded.