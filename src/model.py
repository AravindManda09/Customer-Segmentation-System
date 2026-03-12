import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils import setup_logger

logger = setup_logger(__name__)

class KMeansModel:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, rfm_df):
        logger.info(f"Training KMeans with {self.n_clusters} clusters...")
        
        try:
            rfm_scaled = self.scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
            
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            self.model.fit(rfm_scaled)
            
            labels = self.model.labels_
            rfm_df['Cluster'] = labels
            
            logger.info("Training complete.")
            return rfm_df, labels
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise
            
    def evaluate(self, rfm_df, labels):
        logger.info("Evaluating model...")
        # sanity checks
        if len(rfm_df) != len(labels):
            logger.error(f"Mismatched lengths rfm_df={len(rfm_df)} labels={len(labels)}")
            raise ValueError(f"Number of samples in rfm_df ({len(rfm_df)}) does not match labels ({len(labels)})")
        if len(set(labels)) < 2:
            return 0.0
            
        rfm_scaled = self.scaler.transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
        score = silhouette_score(rfm_scaled, labels)
        
        logger.info(f"Silhouette Score: {score}")
        return score
