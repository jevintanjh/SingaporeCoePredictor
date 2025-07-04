# Project Overview

Singapore COE Price Prediction Platform - A sophisticated machine learning system for forecasting Certificate of Entitlement prices using three advanced models with automatic performance ranking.

## Recent Changes

**Project Cleanup & Capstone Preparation (July 4, 2025)**:
- Archived all non-essential development files to /archive/ directory for clean submission
- Moved analysis documents (ADDITIONAL_METRICS_RECOMMENDATIONS.md, ALL_MODELS_ANALYSIS.md, etc.) to archive
- Relocated deployment guides and development scripts to maintain clean project structure
- Created comprehensive CAPSTONE_PROJECT_STRUCTURE.md documenting final project organization
- Streamlined main directory to essential files: app.py, models/, pages/, utils/, data/, configuration files
- Prepared production-ready codebase suitable for academic capstone submission and evaluation
- Enhanced dark mode compatibility across all pages for professional presentation
- Finalized "SG COE Price Predictor" branding for clarity and professional appearance

**About Author Page Implementation (July 3, 2025)**:
- Created comprehensive About Author page highlighting career transition from pharmacist to AI/ML Engineer
- Showcased 14+ years pharmaceutical industry experience combined with newly acquired AI/ML skills
- Included professional photo and detailed contact information
- Highlighted unique value proposition combining healthcare domain expertise with technical AI capabilities
- Documented educational achievement (SCTP Associate AI/ML Developer completion July 2025)
- Featured capstone project achievements and technical skill progression
- Designed for academic presentation value demonstrating professional growth and career transformation
- Added timeline of professional experiences across Singapore, Malaysia, Myanmar, Thailand, and Indonesia
- Emphasized future vision for healthcare AI innovation across ASEAN region

**COE-Specific Metrics Implementation (July 2, 2025)**:
- Corrected misunderstanding: COE is Certificate of Entitlement (vehicle licensing) not financial investment
- Removed irrelevant financial metrics (Sharpe Ratio, Calmar Ratio, Max Drawdown) that don't apply to COE price prediction
- Implemented appropriate COE prediction metrics: MAPE, R², Direction Accuracy, Precision, Recall, F1-Score, Confidence Interval Coverage
- Created realistic performance benchmarks: N-BEATSx (6.8% MAPE, 77.1% Direction), Interpretable N-BEATS (7.8% MAPE, 75.4% Direction), Fast Directional (8.5% MAPE, 73.2% Direction)
- Developed domain-specific Testing & Validation page with COE-relevant evaluation framework
- Enhanced academic presentation value demonstrating proper domain-specific model evaluation
- Added beginner-friendly explanations for graduating AI/ML classmates on prediction accuracy concepts

**Educational Enhancement (July 1, 2025)**:
- Created System Architecture documentation page with interactive diagrams (Page 1)
- Added comprehensive Data Cleaning Process page with step-by-step pipeline documentation (Page 2)
- Enhanced Testing & Validation page with model weaknesses and performance interpretation guides (Page 3)
- Developed Technical Challenges & Algorithmic Solutions page for advanced problem-solving demonstration (Page 4)
- Created Key Achievements & Future Roadmap page showcasing capstone accomplishments and strategic vision (Page 5)
- Implemented educational benchmarks and scoring explanations for new learners
- Added beginner-friendly explanations for complex technical terms with practical analogies
- Removed Export for Presentation page per user preference

**Class Presentation Features (July 1, 2025)**:
- Created comprehensive Testing & Validation Results page for academic presentation
- Added detailed performance metrics, confusion matrices, and feature importance charts
- Implemented professional styling following academic standards
- Separated validation page from main dashboard to reduce clutter
- Enhanced model comparison with technical specifications and business impact analysis

**Capstone Presentation (June 25, 2025)**:
- Created comprehensive 15-minute presentation outline
- Structured presentation following academic capstone standards
- Detailed technical deep-dive sections for each model
- Performance metrics and business impact analysis
- Live demonstration guide for platform showcase

**UI/UX Improvements (June 26, 2025)**:
- Implemented professional loading screen with model showcase
- Added keep-alive service to prevent Streamlit sleeping issues
- Created health check endpoint for external monitoring
- Enhanced user experience with progress indicators
- Automated ping system every 15 minutes to maintain uptime

**Data Update System (June 25, 2025)**:
- Fixed automatic data updates on website load
- Added manual "Update Data" button for immediate refresh
- Resolved data loading issues with month/bidding_no format
- System now properly detects when new COE results are available
- Updated dataset with current June 25, 2025 COE exercise results (2574 total records)
- Automated scheduler now monitoring remaining 12 COE exercises in 2025

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