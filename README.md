# Singapore COE Price Prediction Platform

## Project Overview

The Singapore COE (Certificate of Entitlement) Price Prediction Platform is a sophisticated machine learning solution designed to forecast COE bidding prices across all vehicle categories. Built with Streamlit, this platform combines advanced time series forecasting models with real-time data integration and intelligent model ranking to provide accurate price predictions for Singapore's unique vehicle quota system.

## Machine Learning Models

The platform employs three distinct forecasting models, each with unique strengths and approaches:

### 1. N-BEATSx Model (Current Top Performer - 92.7% Accuracy)

**Architecture**: Enhanced N-BEATS with exogenous variables
- **Strengths**:
  - Incorporates external market factors (quota sizes, bid counts, seasonal patterns)
  - Excellent at capturing complex market dynamics
  - Superior performance in volatile market conditions
  - Handles multi-variate time series relationships effectively
- **Weaknesses**:
  - Higher computational complexity
  - Requires more historical data to train effectively
  - Less interpretable due to neural network architecture
- **Best Use Case**: Primary predictions during normal market conditions with sufficient historical data

### 2. Interpretable N-BEATS Model (Second Place - 90.6% Accuracy)

**Architecture**: Decomposed N-BEATS with trend and seasonality components
- **Strengths**:
  - Provides interpretable trend and seasonal decomposition
  - Excellent balance between accuracy and explainability
  - Robust performance across different market regimes
  - Clear visualization of underlying patterns
- **Weaknesses**:
  - Slightly lower accuracy than N-BEATSx
  - May miss some complex non-linear relationships
  - Requires parameter tuning for optimal seasonality detection
- **Best Use Case**: When transparency and interpretability are crucial for decision-making

### 3. Fast Directional Forecaster (Third Place - 88.9% Accuracy)

**Architecture**: Lightweight ensemble with momentum indicators
- **Strengths**:
  - Extremely fast training and prediction (sub-second response)
  - Excellent directional accuracy (predicting price increases/decreases)
  - Low computational requirements
  - Robust to data quality issues
- **Weaknesses**:
  - Lower overall price accuracy
  - Simplified feature engineering
  - Less sophisticated pattern recognition
- **Best Use Case**: Quick directional insights and backup predictions when speed is critical

## Unique Platform Features

### Dynamic Model Ranking System

The platform implements an intelligent model ranking system that automatically evaluates and ranks models based on recent performance:

- **Evaluation Criteria**: 70% price accuracy + 30% directional accuracy
- **Time Window**: Focuses on last 6 COE cycles for currency of accuracy
- **Auto-Update**: Rankings refresh automatically when new COE results are published
- **Dynamic UI**: Model tabs reorder automatically with best performer on the left

### Automated Data Updates

**Real-Time Integration**:
- Connects to Singapore Government Open Data API
- Scheduled updates aligned with official COE bidding calendar
- Automatic data validation and duplicate removal
- Fallback web scraping for data reliability

**Update Schedule**:
- Automatically triggered 24 hours after each COE exercise ends
- Manual refresh capability for immediate updates
- Background processing with status monitoring
- Error handling with multiple data source fallbacks

### Advanced Forecasting Features

**Extended Prediction Horizon**:
- Forecasts up to 6 bidding cycles ahead (3 months)
- Confidence intervals for uncertainty quantification
- Category-specific predictions for A, B, C, D, and E categories

**Performance Monitoring**:
- Real-time model performance tracking
- Historical accuracy trends
- Volatility correlation analysis
- Direction prediction accuracy metrics

### Interactive Dashboard

**Multi-Model Comparison**:
- Side-by-side model predictions
- Performance metrics comparison
- Historical accuracy visualization
- Real-time ranking updates

**Data Visualization**:
- Interactive time series plots with Plotly
- Historical trend analysis
- Prediction confidence intervals
- Category-specific insights

**User Experience**:
- Clean, modern interface with responsive design
- Real-time updates without page refresh
- Mobile-friendly layout
- Intuitive navigation with dynamic tab ordering

## Technical Architecture

### Backend Systems
- **Data Processing**: Pandas and NumPy for efficient data manipulation
- **Model Training**: Scikit-learn with custom ensemble methods
- **Scheduling**: Background task scheduler for automated updates
- **Validation**: Cross-validation with walk-forward analysis

### Frontend Interface
- **Framework**: Streamlit for rapid deployment
- **Visualization**: Plotly for interactive charts
- **Styling**: Custom CSS for modern UI/UX
- **Performance**: Caching strategies for optimal response times

### Data Pipeline
- **Sources**: Singapore Government API, web scraping fallbacks
- **Processing**: Automated cleaning, validation, and feature engineering
- **Storage**: CSV-based with backup systems
- **Quality Control**: Duplicate detection and anomaly filtering

## Market Impact and Applications

### For Individual Buyers
- Informed bidding decisions with 6-cycle forecasts
- Category comparison for optimal timing
- Confidence intervals for risk assessment

### For Industry Professionals
- Market trend analysis and reporting
- Strategic planning with extended forecasts
- Performance benchmarking across models

### For Researchers and Analysts
- Model interpretability features for market understanding
- Historical data access for research purposes
- API-ready architecture for integration

## Performance Validation

The platform employs rigorous validation methodologies:

- **Walk-Forward Validation**: Simulates real-world prediction scenarios
- **Bootstrap Testing**: Assesses model stability across data variations
- **Regime Testing**: Validates performance across different market conditions
- **Cross-Model Validation**: Ensemble approach for improved reliability

## Future Enhancements

- **Real-Time Bidding Integration**: Live updates during bidding exercises
- **Mobile Application**: Dedicated mobile app for on-the-go access
- **Advanced Analytics**: Market sentiment analysis and external factor integration
- **API Services**: RESTful API for third-party integrations

## Conclusion

This COE prediction platform represents a comprehensive solution for Singapore's unique vehicle quota market, combining state-of-the-art machine learning with practical user needs. The multi-model approach ensures robust predictions while the automated ranking system maintains optimal performance as market conditions evolve. With its focus on accuracy, interpretability, and user experience, the platform serves as a valuable tool for navigating Singapore's COE bidding landscape.
