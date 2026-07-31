# test_app.py
import os
import pytest
from app import load_and_sync_samples

def test_sample_generation_pipeline():
    """
    Verifies that the texture asset generator initializes 
    the local directory structure correctly.
    """
    config_file = "samples.config"
    created_mock = False
    
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write("mock_cat.jpg,cat\nmock_car.jpg,car\n")
        created_mock = True
        
    manifest = load_and_sync_samples()
    
    assert isinstance(manifest, list), "Manifest must return a list layout structure."
    assert os.path.exists("raw_unorganized_files"), "Target asset container directory missing."
    
    if created_mock and os.path.exists(config_file):
        os.remove(config_file)
