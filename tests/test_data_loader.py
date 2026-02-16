import pytest
import os
import pandas as pd
from src.data_loader import generate_transactions, load_data, save_data

def test_generate_transactions():
    df = generate_transactions(n_rows=50, n_customers=5)
    assert not df.empty
    assert len(df) == 50
    assert 'CustomerID' in df.columns
    assert 'Amount' in df.columns

def test_save_and_load_data(tmp_path):
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    file_path = tmp_path / "test_data.csv"
    save_data(df, str(file_path))
    
    assert os.path.exists(file_path)
    
    loaded_df = load_data(str(file_path))
    pd.testing.assert_frame_equal(df, loaded_df)
