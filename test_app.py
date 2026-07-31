import os
import sys
from unittest.mock import MagicMock

# Create a fake Streamlit mock object to stop decorators from crashing the test environment
mock_st = MagicMock()
mock_st.cache_resource = lambda x=None, *args, **kwargs: x if callable(x) else lambda f: f

# Inject the mock directly into the active Python system modules
sys.modules['streamlit'] = mock_st

# NOW it is completely safe to import your app code without triggering Streamlit errors
import pytest
from app import load_and_sync_samples

def test_sample_generation_pipeline():
    """
    Verifies that the texture asset generator initializes 
    the local directory structure correctly.
    """
    # 1. Setup a mock samples.config if it doesn't exist
    config_file = "samples.config"
    created_mock = False
    
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write("mock_cat.jpg,cat\nmock_car.jpg,car\n")
        created_mock = True
        
    # 2. Run the core generation routine
    manifest = load_and_sync_samples()
    
    # 3. Assert structural compliance
    assert isinstance(manifest, list), "Manifest must return a list layout structure."
    assert os.path.exists("raw_unorganized_files"), "Target asset container directory missing."
    
    # 4. Clean up mock files if they were created during the test loop
    if created_mock:
        if os.path.exists(config_file):
            os.remove(config_file)
