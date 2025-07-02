# Streamlit Cloud entry point
# This file ensures compatibility with Streamlit Cloud deployment

import streamlit as st
import os

# Set environment variable to indicate Streamlit Cloud deployment
os.environ['STREAMLIT_CLOUD_DEPLOYMENT'] = 'true'

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()