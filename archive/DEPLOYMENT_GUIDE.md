# Streamlit Cloud Deployment Guide

## Files Required for Streamlit Cloud

### 1. Main App Files
- `app.py` - Main application entry point
- `streamlit_app.py` - Streamlit Cloud specific entry point
- `.streamlit/config.toml` - Streamlit configuration

### 2. Configuration Files
```toml
# .streamlit/config.toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### 3. Requirements (already in requirements.txt)
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- trafilatura
- schedule
- requests

### 4. Environment Detection
The app automatically detects if running on Streamlit Cloud and disables:
- Keep-alive service (only needed for Replit)
- Port-specific configurations

### 5. Deployment Steps
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Use `streamlit_app.py` as entry point
4. Deploy with default settings

### 6. Features Available on Streamlit Cloud
✅ All 3 ML models (N-BEATSx, Interpretable N-BEATS, Fast Directional)
✅ Real-time predictions and historical analysis
✅ Educational pages (System Architecture, Data Cleaning, etc.)
✅ Model performance ranking
✅ Interactive visualizations

### 7. Features Disabled on Streamlit Cloud
❌ Keep-alive service (not needed on cloud)
❌ Local port configuration
❌ Replit-specific settings

## Troubleshooting

### Connection Refused Error
If you see "connection refused" errors, ensure:
1. Keep-alive service is disabled for cloud deployment
2. No hardcoded port references
3. Using `streamlit_app.py` as entry point

### Import Errors
Ensure all dependencies are in requirements.txt with compatible versions.

### Performance Issues
The app loads 2,574+ historical records on startup. First load may take 10-15 seconds.