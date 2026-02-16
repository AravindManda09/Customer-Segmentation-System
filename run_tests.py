import sys
import pytest
import os

# Ensure src is in path
sys.path.append(os.getcwd())

if __name__ == '__main__':
    # Run pytest
    exit_code = pytest.main(["-v", "tests/"])
    sys.exit(exit_code)
