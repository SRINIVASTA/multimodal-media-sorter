# conftest.py
import sys
from unittest.mock import MagicMock

# 1. Intercept and mock out the Streamlit rendering components globally
mock_st = MagicMock()
mock_st.cache_resource = lambda x=None, *args, **kwargs: x if callable(x) else lambda f: f
sys.modules['streamlit'] = mock_st

# 2. Mock out sentence-transformers to prevent heavy weight downloads during test scans
mock_transformers = MagicMock()
mock_transformers.SentenceTransformer = lambda *args, **kwargs: MagicMock()
sys.modules['sentence_transformers'] = mock_transformers
