import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.utils import setup_logger

logger = setup_logger(__name__)

def plot_rfm_distribution(rfm_df):
    try:
        logger.info("Plotting RFM distribution...")
        fig_r = px.histogram(rfm_df, x='Recency', title='Recency Distribution')
        fig_f = px.histogram(rfm_df, x='Frequency', title='Frequency Distribution')
        fig_m = px.histogram(rfm_df, x='Monetary', title='Monetary Distribution')
        return fig_r, fig_f, fig_m
    except Exception as e:
        logger.error(f"Error plotting RFM distribution: {e}")
        return None, None, None

def plot_3d_clusters(rfm_df):
    try:
        logger.info("Plotting 3D Clusters...")
        fig = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary',
                            color='Cluster', title='3D Cluster Visualization',
                            opacity=0.7, symbol='Cluster')
        return fig
    except Exception as e:
        logger.error(f"Error plotting 3D clusters: {e}")
        return None

def plot_cluster_summary(rfm_df):
    try:
        logger.info("Plotting Cluster Summary...")
        avg_df = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=avg_df['Cluster'], y=avg_df['Recency'], name='Avg Recency'))
        fig.add_trace(go.Bar(x=avg_df['Cluster'], y=avg_df['Frequency'], name='Avg Frequency'))
        fig.add_trace(go.Bar(x=avg_df['Cluster'], y=avg_df['Monetary'], name='Avg Monetary'))
        
        fig.update_layout(barmode='group', title='Average RFM Values by Cluster')
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting cluster summary: {e}")
        return None
