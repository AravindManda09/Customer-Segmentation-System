import pytest
import pandas as pd
from src.model import KMeansModel

@pytest.fixture
def sample_rfm():
    return pd.DataFrame({
        'Recency': [10, 20, 30, 40, 50, 10, 5],
        'Frequency': [5, 4, 3, 2, 1, 5, 6],
        'Monetary': [100, 80, 60, 40, 20, 100, 120]
    })

def test_kmeans_training(sample_rfm):
    model = KMeansModel(n_clusters=2)
    rfm_df, labels = model.train(sample_rfm)
    
    assert 'Cluster' in rfm_df.columns
    assert len(labels) == len(sample_rfm)
    assert rfm_df['Cluster'].nunique() <= 2

def test_kmeans_evaluation(sample_rfm):
    model = KMeansModel(n_clusters=2)
    rfm_df, labels = model.train(sample_rfm)
    score = model.evaluate(rfm_df, labels)
    
    # Silhouette score is between -1 and 1
    assert -1 <= score <= 1
