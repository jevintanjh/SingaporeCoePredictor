# Project Overview

Singapore COE Price Prediction Platform - A sophisticated machine learning system for forecasting Certificate of Entitlement prices using three advanced models with automatic performance ranking.

## Recent Changes

**Data Update System (June 25, 2025)**:
- Fixed automatic data updates on website load
- Added manual "Update Data" button for immediate refresh
- Resolved data loading issues with month/bidding_no format
- System now properly detects when new COE results are available

**Codebase Cleanup**:
- Removed 17 unused model files to streamline project
- Fixed Streamlit Cloud deployment by removing TensorFlow dependencies
- Created comprehensive capstone compliance documentation

**Model Performance (Last 6 Cycles)**:
- N-BEATSx: 92.7% accuracy (Rank #1)
- Interpretable N-BEATS: 90.6% accuracy (Rank #2) 
- Fast Directional Forecaster: 88.9% accuracy (Rank #3)

## Project Architecture

**Core Models**: 3 production models in /models/ directory
**Data Pipeline**: Automated updates from Singapore Government API
**Ranking System**: Dynamic performance-based model ordering
**Frontend**: Streamlit dashboard with interactive visualizations
**Deployment**: Ready for Streamlit Cloud with proper requirements.txt

## User Preferences

**Communication Style**: Clear, concise explanations without excessive technical jargon
**Update Frequency**: Automatic data checks on load, manual refresh available
**Model Display**: Dynamic tab ordering based on recent performance rankings
**Documentation**: Comprehensive but accessible for non-technical users

## Technical Notes

**Data Format**: CSV with month/bidding_no columns converted to date format
**Update Logic**: Checks for data older than 3 days, fetches from government API
**Model Evaluation**: Walk-forward validation with 70% price + 30% directional accuracy
**Deployment**: Streamlit Cloud compatible, no TensorFlow dependencies required