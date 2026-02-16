import pandas as pd
import datetime as dt
from src.utils import setup_logger

logger = setup_logger(__name__)

def calculate_rfm(df, config=None):
    """
    Calculates RFM metrics.
    """
    logger.info("Calculating RFM metrics...")
    
    # Use config if provided, else defaults
    cust_col = 'CustomerID'
    date_col = 'TransactionDate'
    amt_col = 'Amount'
    
    if config:
        # If column names were in config, we'd use them here. 
        # For now, assuming standard names based on generation.
        pass

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        max_date = df[date_col].max() + dt.timedelta(days=1)
        
        rfm = df.groupby(cust_col).agg({
            date_col: lambda x: (max_date - x.max()).days,
            cust_col: 'count',
            amt_col: 'sum'
        }).rename(columns={
            date_col: 'Recency',
            cust_col: 'Frequency',
            amt_col: 'Monetary'
        })
        
        logger.info(f"RFM calculation complete for {len(rfm)} customers.")
        return rfm
        
    except Exception as e:
        logger.error(f"Error in RFM calculation: {e}")
        raise
