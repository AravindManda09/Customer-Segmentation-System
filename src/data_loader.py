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

def load_data(file_source):
    """
    Loads data from a CSV file path or buffer. 
    Handles comma/tab separators and ensures data is loaded generically.
    """
    logger.info(f"Loading data...")
    
    # Check string path
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            logger.error(f"File not found: {file_source}")
            raise FileNotFoundError(f"File not found: {file_source}")
    
    try:
        encoding_used = 'utf-8'
        # 1. Try default load (comma) with UTF-8
        if hasattr(file_source, 'seek'): file_source.seek(0)
        try:
            df = pd.read_csv(file_source)
        except UnicodeDecodeError:
            logger.info("UTF-8 decoding failed. Trying UTF-16...")
            if hasattr(file_source, 'seek'): file_source.seek(0)
            df = pd.read_csv(file_source, encoding='utf-16')
            encoding_used = 'utf-16'
            
        # 2. Check for single column (likely wrong separator)
        if len(df.columns) == 1:
            logger.info("Single column detected. Trying tab separator...")
            if hasattr(file_source, 'seek'): file_source.seek(0)
            try:
                df_tab = pd.read_csv(file_source, sep='\\t', encoding=encoding_used)
                if len(df_tab.columns) > 1:
                    df = df_tab
            except Exception:
                pass
        
        # 3. Check for obvious headerless data where the first row is purely numeric/date
        # If the first column's name looks like a number or ID instead of a feature name
        # We can optionally drop down to header=None, but generally we will assume 
        # the user uploaded a CSV with headers. We no longer force ['CustomerID', 'TransactionDate', 'Amount', 'OrderID'].
        
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def save_data(df, file_path):
    """
    Saves dataframe to CSV.
    """
    logger.info(f"Saving data to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info("Data saved successfully.")
