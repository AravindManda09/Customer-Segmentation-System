import streamlit as st
import pandas as pd
from src.utils import load_config, setup_logger
from src.data_loader import generate_transactions, load_data
from src.preprocessor import calculate_rfm
from src.model import KMeansModel
from src.visualization import plot_rfm_distribution, plot_3d_clusters, plot_cluster_summary

# Setup
logger = setup_logger("app")
config = load_config()

st.set_page_config(page_title="Customer Segmentation System", layout="wide")
st.title("Customer Segmentation System Using RFM & K-Means")

# Sidebar
st.sidebar.header("Data Configuration")
data_source = st.sidebar.radio("Select Data Source", ["Generate Synthetic Data", "Upload CSV"])

df = pd.DataFrame()

try:
    if data_source == "Generate Synthetic Data":
        n_rows = st.sidebar.slider("Number of Transactions", 100, 5000, config['data_generation']['n_rows'])
        n_customers = st.sidebar.slider("Number of Customers", 50, 500, config['data_generation']['n_customers'])
        
        if st.sidebar.button("Generate Data"):
            df = generate_transactions(n_rows=n_rows, n_customers=n_customers, random_state=config['data_generation']['random_state'])
            st.session_state['df'] = df
            logger.info("Data generated via UI.")
            
        elif 'df' in st.session_state:
            df = st.session_state['df']
            
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your transaction CSV", type=["csv", "tsv", "txt"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.session_state['df'] = df
            logger.info("Data uploaded via UI.")

    # Main Processing
    if not df.empty:
        tab1, tab2, tab3 = st.tabs(["Data Preview", "RFM Analysis", "Clustering & Insights"])
        
        with tab1:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            st.write(f"Total Rows: {len(df)}")
            st.write(f"Total Customers: {df[st.session_state.get('cust_col', df.columns[0])].nunique() if 'cust_col' in st.session_state else 'N/A'}")
            
        st.sidebar.subheader("Map Your Columns")
        cols = list(df.columns)
        
        cust_col = st.sidebar.selectbox("Customer ID Column", cols, index=0)
        date_col = st.sidebar.selectbox("Transaction Date Column", cols, index=1 if len(cols) > 1 else 0)
        amt_col = st.sidebar.selectbox("Transaction Amount Column", cols, index=2 if len(cols) > 2 else 0)
        
        st.session_state['cust_col'] = cust_col
        
        # Build config dictionary for preprocessor
        rfm_config = {
            'cust_col': cust_col,
            'date_col': date_col,
            'amt_col': amt_col
        }
            
        try:
            rfm_df = calculate_rfm(df, config=rfm_config)
            
            with tab2:
                st.subheader("RFM Analysis")
                if len(rfm_df) > 0:
                    st.dataframe(rfm_df.head())
                    
                    st.subheader("Distributions")
                    fig_r, fig_f, fig_m = plot_rfm_distribution(rfm_df)
                    col1, col2, col3 = st.columns(3)
                    with col1: st.plotly_chart(fig_r, use_container_width=True)
                    with col2: st.plotly_chart(fig_f, use_container_width=True)
                    with col3: st.plotly_chart(fig_m, use_container_width=True)
                else:
                    st.warning("RFM DataFrame is empty. Please check your column mappings.")
                    
            if rfm_df is not None and not rfm_df.empty:
                with tab3:
                    st.subheader("K-Means Clustering")
                    n_clusters = st.slider("Select Number of Clusters", 2, 10, config['model']['n_clusters'])
                    
                    if st.button("Run Clustering"):
                        model = KMeansModel(n_clusters=n_clusters, random_state=config['model']['random_state'])
                        rfm_clustered, labels = model.train(rfm_df)
                        st.session_state['rfm_clustered'] = rfm_clustered
                        st.session_state['model_obj'] = model
                        
                    if 'rfm_clustered' in st.session_state:
                        rfm_cl = st.session_state['rfm_clustered']
                        model_obj = st.session_state.get('model_obj') # Retrieval
                        
                        # Metrics
                        if model_obj:
                            score = model_obj.evaluate(rfm_df, rfm_cl['Cluster'])
                            st.info(f"Silhouette Score: {score:.3f}")
                        
                        # Visualizations
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.plotly_chart(plot_3d_clusters(rfm_cl), use_container_width=True)
                        with col2:
                            st.plotly_chart(plot_cluster_summary(rfm_cl), use_container_width=True)
                            
                        # Insights logic remains similar...
                        st.subheader("Cluster Insights")
                        avg_values = rfm_cl.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
                        st.table(avg_values)
                        
        except Exception as e:
            st.error(f"Error calculating RFM. Please ensure you selected a valid Date column and a numeric Amount column. Details: {e}")
            rfm_df = None

except Exception as e:
    st.error(f"An error occurred: {e}")
    logger.error(f"Global error: {e}")

else:
    if df.empty:
        st.info("Please generate or upload data to begin.")
