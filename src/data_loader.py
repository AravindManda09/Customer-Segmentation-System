import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
from src.utils import setup_logger

logger = setup_logger(__name__)

def generate_transactions(n_rows=1000, n_customers=100, random_state=42):
    """
    Generates synthetic transaction data.
    """
    logger.info(f"Generating {n_rows} transactions for {n_customers} customers.")
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
        
    customer_ids = [f'CUST_{i:04d}' for i in range(1, n_customers + 1)]
    
    data = []
    
    for _ in range(n_rows):
        customer_id = random.choice(customer_ids)
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        transaction_date = start_date + timedelta(days=random_days)
        
        amount = round(np.random.gamma(shape=2, scale=50), 2)
        amount = max(5.0, amount)
        
        order_id = f'ORD_{random.randint(10000, 99999)}'
        
        data.append([customer_id, transaction_date, amount, order_id])
        
    df = pd.DataFrame(data, columns=['CustomerID', 'TransactionDate', 'Amount', 'OrderID'])
    df = df.sort_values('TransactionDate').reset_index(drop=True)
    
    logger.info("Data generation complete.")
    return df

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows.")
    return df

def save_data(df, file_path):
    """
    Saves dataframe to CSV.
    """
    logger.info(f"Saving data to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info("Data saved successfully.")
