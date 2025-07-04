# Singapore COE Price Prediction Platform - Capstone Project Structure

## Project Overview
**Author:** Tan Jue Hao Jevin  
**Program:** SkillsFuture Career Transition Programme (SCTP) - Associate AI/ML Developer  
**Institution:** NTUC Learning Hub  
**Completion:** July 2025  

## Executive Summary
Advanced machine learning platform for forecasting Singapore Certificate of Entitlement (COE) prices using three distinct models with dynamic performance ranking and real-time data integration.

## Core Project Files

### 🚀 Main Application
- **`app.py`** - Primary Streamlit application with model integration and dashboard
- **`streamlit_app.py`** - Streamlit Cloud deployment entry point
- **`requirements.txt`** - Python dependencies for deployment
- **`README.md`** - Project documentation and setup instructions
- **`replit.md`** - Technical architecture and user preferences documentation

### 🤖 Machine Learning Models (`/models/`)
- **`fast_directional_forecaster.py`** - High-speed momentum analysis model (88.9% accuracy)
- **`interpretable_nbeats_v2.py`** - Transparent N-BEATS implementation (90.6% accuracy)
- **`nbeatsx.py`** - Advanced neural forecasting with exogenous variables (92.7% accuracy)

### 📊 Interactive Pages (`/pages/`)
- **`1_System_Architecture.py`** - Educational system design documentation
- **`2_Data_Cleaning_Process.py`** - Comprehensive data pipeline explanation
- **`3_Testing_and_Validation.py`** - Model performance analysis and comparison
- **`4_Technical_Challenges.py`** - Advanced algorithmic solutions showcase
- **`5_Achievements_and_Future.py`** - Project accomplishments and roadmap
- **`6_About_Author.py`** - Professional profile and career transition story

### 🔧 Utilities (`/utils/`)
- **`data_updater.py`** - Real-time COE data fetching from Singapore Government API
- **`coe_scheduler.py`** - Automated bidding schedule monitoring and updates
- **`model_ranker.py`** - Dynamic performance-based model ranking system
- **`financial_evaluator.py`** - COE-specific evaluation metrics (MAPE, Direction Accuracy)

### 📈 Data (`/data/`)
- **`merged_coe_data.csv`** - Comprehensive historical COE dataset (2002-2025, 2,574 records)
- **`bidding_schedule.json`** - Automated schedule tracking for 2025 exercises

### ⚙️ Configuration
- **`.streamlit/config.toml`** - Streamlit server configuration
- **`.replit`** - Replit deployment and workflow configuration
- **`health_check.py`** - Application monitoring endpoint
- **`keep_alive.py`** - Anti-sleep service for continuous operation

## Key Technical Achievements

### 🎯 Machine Learning Excellence
- **Three Advanced Models:** Fast Directional, Interpretable N-BEATS, N-BEATSx
- **Dynamic Ranking:** Performance-based model selection (accuracy rates 88.9%-92.7%)
- **Real-time Predictions:** 6 bidding cycles ahead with confidence intervals
- **Rigorous Validation:** Walk-forward testing and bootstrap validation

### 🔄 Data Engineering
- **Automated Pipeline:** Singapore Government API integration with scheduling
- **Real-time Updates:** Automatic detection and incorporation of new COE results
- **Data Quality:** Comprehensive cleaning, validation, and standardization
- **Historical Analysis:** 23+ years of COE bidding data (2002-2025)

### 🎨 User Experience
- **Interactive Dashboard:** Multi-model comparison with performance ranking
- **Educational Content:** Six dedicated pages for academic presentation
- **Professional Design:** Dark/light mode compatibility with responsive layout
- **Performance Monitoring:** Real-time metrics and model comparison matrices

### 🚀 Deployment Ready
- **Cloud Compatible:** Streamlit Cloud deployment with proper configuration
- **Scalable Architecture:** Modular design for easy maintenance and updates
- **Production Monitoring:** Health checks and automated keep-alive services
- **Documentation:** Comprehensive technical and user documentation

## Academic Value

### 📚 Learning Objectives Demonstrated
- **Time Series Forecasting:** Advanced neural and statistical approaches
- **Model Interpretability:** Transparent AI for regulatory compliance
- **Performance Evaluation:** Domain-specific metrics (MAPE, Direction Accuracy)
- **Data Engineering:** Real-time pipeline with government API integration
- **Web Development:** Full-stack ML application with interactive visualization

### 🎓 Capstone Compliance
- **Industry Problem:** Real Singapore vehicle quota pricing challenge
- **Technical Depth:** Three distinct ML approaches with comparative analysis
- **Practical Application:** Production-ready deployment with real data
- **Educational Design:** Beginner-friendly explanations for newly graduated AI/ML peers
- **Professional Presentation:** Clean structure suitable for academic showcase

## Archive Directory (`/archive/`)
Contains development artifacts, analysis documents, and non-essential files moved during project cleanup for submission.

---

**Note:** This capstone project demonstrates the complete AI/ML development lifecycle from data acquisition through model deployment, specifically tailored for Singapore's COE prediction domain with educational value for newly graduated AI/ML professionals.