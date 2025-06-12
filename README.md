# Singapore COE Price Prediction Platform

A sophisticated machine learning platform for predicting Singapore's Certificate of Entitlement (COE) bidding prices across all vehicle categories.

## Features

- **Multi-Model Forecasting**: Three advanced ML models with automatic performance ranking
- **Real-Time Updates**: Automated data synchronization with Singapore Government APIs
- **Extended Predictions**: Forecasts up to 6 bidding cycles ahead
- **Interactive Dashboard**: Dynamic visualizations with Plotly
- **Smart Ranking**: Models automatically reorder based on recent performance

## Models

1. **N-BEATSx** (Current leader: 92.7% accuracy) - Advanced neural forecasting with external market factors
2. **Interpretable N-BEATS** (90.6% accuracy) - Transparent forecasting with trend decomposition
3. **Fast Directional Forecaster** (88.9% accuracy) - Lightweight model optimized for speed

## Quick Start

### Local Development

```bash
git clone <your-repo-url>
cd coe-prediction-platform
pip install -r streamlit_requirements.txt
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository
5. Set `app.py` as the main file

## Data Sources

- Singapore Government Open Data API
- Land Transport Authority (LTA) official sources
- Automated web scraping fallbacks

## Technical Stack

- **Backend**: Python, Pandas, NumPy, Scikit-learn
- **Frontend**: Streamlit, Plotly
- **Data**: Real-time government APIs
- **ML**: Custom ensemble methods with validation

## Usage

The platform automatically:
- Updates data after each COE exercise
- Ranks models by recent performance
- Provides confidence intervals for predictions
- Displays historical accuracy trends

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests welcome for model improvements and feature enhancements.