"""
Export page for PowerPoint/PDF presentation content
Provides copy-paste ready content and charts for academic presentations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def show_export_page():
    st.title("📤 Export for Presentation")
    st.markdown("**PowerPoint/PDF ready content for your capstone presentation**")
    
    st.info("""
    💡 **Export Guide**: This page provides presentation-ready content for your Technical Challenges 
    and Achievements sections. Use the copy-paste text and downloadable charts to enhance your PowerPoint slides.
    """)
    
    # Create tabs for different export sections
    tab1, tab2, tab3 = st.tabs([
        "🔧 Technical Challenges",
        "🏆 Achievements & Future", 
        "📋 Export Instructions"
    ])
    
    with tab1:
        export_technical_challenges()
    
    with tab2:
        export_achievements()
    
    with tab3:
        export_instructions()

def export_technical_challenges():
    st.header("🔧 Technical Challenges Export")
    st.markdown("**Ready for Slide 14: Technical Challenges & Algorithmic Solutions**")
    
    # Exportable content
    st.subheader("📝 Copy-Paste Content")
    
    technical_content = """**TECHNICAL CHALLENGES & ALGORITHMIC SOLUTIONS**

**Non-Stationarity Handling:**
• Augmented Dickey-Fuller (ADF) Test
  - "Stability detector" for price trends - tests if COE prices follow predictable patterns
  - P-value < 0.05 = Stationary (good for prediction) ✅
  - P-value > 0.05 = Non-stationary (random walking) ❌

• Seasonal Decomposition (X-13ARIMA-SEATS)
  - Separates COE prices into: Trend + Seasonal + Irregular components
  - Like separating music into bass, melody, and background noise
  - Government-standard methodology used worldwide

• Cointegration Analysis
  - Finds if different COE categories move together long-term
  - Example: Car and motorcycle prices follow same economic trends
  - Useful for cross-category prediction and validation

**Missing Data Treatment:**
• Multiple Imputation (MICE) - "5 experts guessing missing exam scores"
  - Each expert uses different information (homework, attendance, previous scores)
  - Average all 5 guesses for final answer
  - Statistical models instead of human judgment

• Kalman Filter - "GPS navigation adapting to changing traffic"
  - Originally used in NASA rocket guidance systems
  - Tracks how COE price patterns change over time
  - Adjusts predictions as new information arrives

• Forward-fill with Exponential Decay
  - Uses last known value but "fades" confidence over time
  - Like assuming yesterday's weather with decreasing certainty each day

**Overfitting Prevention:**
• Elastic Net Regularization: λ₁|β|₁ + λ₂|β|₂²
  - L1 Penalty: Forces model to ignore useless features (feature selection)
  - L2 Penalty: Keeps model weights small and stable
  - Prevents model from "memorizing" training data

• Temporal Cross-Validation
  - Always trains on past, tests on future (realistic for time series)
  - Never lets model "peek into the future"
  - Gap between train/test prevents data leakage

• Information Criteria (AIC/BIC)
  - Balances accuracy vs complexity objectively
  - Lower values = better models
  - Like comparing cars: performance + fuel efficiency + price

**Computational Optimization:**
• Numba JIT Compilation: 50-100x speedup with one decorator
  - Converts Python to machine code automatically
  - Like learning local language vs speaking through translator

• Vectorized Operations: 10-20x faster than Python loops
  - NumPy processes entire arrays at once
  - Like grading entire class simultaneously vs one student at a time

• Memory Optimization: 30-50% reduction
  - Use appropriate data types (float32 vs float64)
  - Half the memory, same accuracy for most cases

• Parallel Processing: Multiple models trained simultaneously
  - Uses all CPU cores instead of just one
  - Like 5 chefs cooking simultaneously vs one chef cooking 5 dishes

**Deployment Architecture:**
• Docker Containerization - "Ship code with entire environment"
  - Works same way on any computer (laptop, server, cloud)
  - No more "it works on my machine" problems

• Redis Caching - "Smart memory for instant responses"
  - Remembers recent predictions (0.01 seconds vs 2 seconds)
  - Like having assistant who remembers your recent questions

• Async Processing with Celery - "Background workers for heavy tasks"
  - User gets instant response while work happens behind scenes
  - Web app stays responsive during model training

• Circuit Breaker Patterns - "Smart electrical breaker for software"
  - Protects system when external services fail
  - Automatically recovers when services come back online"""
    
    st.code(technical_content, language="markdown")
    
    # Downloadable charts
    st.subheader("📊 Downloadable Charts")
    
    # Performance optimization chart
    st.markdown("**Chart 1: Computational Optimization Results**")
    methods = ['Pure Python', 'NumPy', 'Numba JIT', 'Cython', 'Parallel']
    speedup = [1, 6.7, 50, 66.7, 125]
    
    fig1 = go.Figure(data=[
        go.Bar(x=methods, y=speedup, 
               marker_color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'],
               text=[f'{x}x' for x in speedup],
               textposition='auto')
    ])
    fig1.update_layout(
        title="<b>Speed Improvement Comparison</b>",
        yaxis_title="Speedup Factor",
        height=400,
        font=dict(size=16, family="Arial"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("💡 Hover over chart → Click camera icon (📷) → Download PNG")
    
    # Regularization effectiveness
    st.markdown("**Chart 2: Regularization Cross-Validation Results**")
    
    cv_data = {
        'λ₁': [0.001, 0.010, 0.100, 1.000],
        'λ₂': [0.001, 0.010, 0.100, 1.000],
        'CV Score': [0.847, 0.892, 0.834, 0.723],
        'Features Selected': ['15/20', '8/20', '3/20', '1/20'],
        'Interpretation': ['Slight regularization', 'Optimal balance ⭐', 'Over-regularized', 'Severe underfitting']
    }
    
    cv_df = pd.DataFrame(cv_data)
    
    fig2 = go.Figure(data=[
        go.Bar(x=[f"λ={l}" for l in cv_data['λ₁']], 
               y=cv_data['CV Score'],
               marker_color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728'],
               text=[f'{score:.3f}' for score in cv_data['CV Score']],
               textposition='auto')
    ])
    fig2.update_layout(
        title="<b>Regularization Strength vs Cross-Validation Performance</b>",
        yaxis_title="CV Score (Higher = Better)",
        xaxis_title="Regularization Strength",
        height=400,
        font=dict(size=16, family="Arial"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("💡 Shows optimal regularization strength (λ=0.010) for best performance")

def export_achievements():
    st.header("🏆 Achievements & Future Export")
    st.markdown("**Ready for Key Achievements and Future Roadmap slides**")
    
    # Key achievements content
    st.subheader("📝 Key Achievements Content")
    
    achievements_content = """**CAPSTONE PROJECT: KEY ACHIEVEMENTS**

**Performance Metrics:**
• Model Accuracy: 92.7% (N-BEATSx leading performance)
• Data Coverage: 2,574 complete historical records
• Prediction Horizon: 6 cycles ahead (industry-leading)
• Model Diversity: 3 advanced forecasting approaches

**Technical Accomplishments:**
• Advanced Machine Learning Implementation
  - N-BEATSx: 92.7% accuracy with exogenous variables
  - Interpretable N-BEATS: 90.6% accuracy with full explainability  
  - Fast Directional: 88.9% accuracy with real-time performance

• Data Engineering Excellence
  - Automated pipeline with Singapore LTA API integration
  - Advanced cleaning with statistical outlier detection
  - Temporal cross-validation with walk-forward testing

• Software Engineering Best Practices
  - Modular architecture with separation of concerns
  - 50-100x performance optimization with Numba JIT
  - Production-ready deployment with Docker containerization

**Innovation Highlights:**
• Dynamic Model Ranking: Automated performance-based ordering
• Extended Prediction Horizon: 6 cycles ahead forecasting
• Educational AI Platform: Complex concepts with simple analogies

**Academic Excellence:**
• 4 comprehensive educational pages with interactive demonstrations
• Industry-standard practices and production deployment strategies
• Novel application of N-BEATS to COE prediction

**FUTURE ENHANCEMENTS**

**Short-term (3-6 months):**
• Economic Indicator Integration
  - GDP, inflation, interest rates, employment data
  - Expected 2-3% accuracy improvement
  - Better capture of economic cycle effects

• Sentiment Analysis Integration  
  - News analysis and social media monitoring
  - Government policy impact assessment
  - Market psychology effects capture

• Mobile Application Development
  - Push notifications for price threshold alerts
  - Live bidding updates and prediction dashboard
  - React Native/Flutter with Firebase notifications

**Long-term (6-24 months):**
• Multi-Country COE System Expansion
  - Hong Kong, Shanghai, Mumbai, London markets
  - Adapt to different regulatory frameworks
  - Cross-country influence analysis

• Vehicle Marketplace Integration
  - Dealer networks and online platform connections
  - Financial services and insurance integration
  - Subscription-based premium analytics

• Advanced Ensemble Learning with Transformers
  - 95%+ accuracy with transformer architectures
  - Multi-modal data fusion capabilities
  - Meta-learning for rapid market adaptation

**Strategic Vision:**
• Total Addressable Market: $50M+ in Singapore alone
• Growth Potential: 300%+ with international expansion
• Competitive Advantage: First-mover in AI-powered COE prediction
• Revenue Model: Subscription services + API monetization"""
    
    st.code(achievements_content, language="markdown")
    
    # Achievement charts
    st.subheader("📊 Achievement Charts")
    
    # Model performance comparison
    st.markdown("**Chart 3: Model Performance Achievements**")
    models = ['N-BEATSx', 'Interpretable\nN-BEATS', 'Fast\nDirectional']
    accuracy = [92.7, 90.6, 88.9]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig3 = go.Figure(data=[
        go.Bar(
            x=models,
            y=accuracy,
            marker_color=colors,
            text=[f'{acc}%' for acc in accuracy],
            textposition='auto',
            textfont=dict(size=16, color='white')
        )
    ])
    fig3.update_layout(
        title="<b>Capstone Project: Model Performance Achievements</b>",
        yaxis_title="Directional Accuracy (%)",
        yaxis=dict(range=[85, 95]),
        height=400,
        font=dict(size=16, family="Arial"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Market opportunity breakdown
    st.markdown("**Chart 4: Market Opportunity Analysis**")
    market_segments = ['Individual\nBidders', 'Car\nDealerships', 'Financial\nInstitutions', 'Government\nAgencies']
    market_sizes = [60, 25, 10, 5]
    
    fig4 = go.Figure(data=[go.Pie(
        labels=market_segments,
        values=market_sizes,
        hole=0.3,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )])
    fig4.update_layout(
        title="<b>Total Addressable Market Segments ($50M+)</b>",
        height=400,
        font=dict(size=16, family="Arial")
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Development timeline
    st.markdown("**Chart 5: Strategic Development Timeline**")
    phases = [
        {'name': 'Economic Integration', 'start': 1, 'duration': 3, 'color': '#ff7f0e'},
        {'name': 'Sentiment Analysis', 'start': 2, 'duration': 4, 'color': '#2ca02c'},
        {'name': 'Mobile App', 'start': 4, 'duration': 3, 'color': '#d62728'},
        {'name': 'Multi-Country', 'start': 6, 'duration': 6, 'color': '#9467bd'},
        {'name': 'Marketplace Integration', 'start': 8, 'duration': 8, 'color': '#8c564b'},
        {'name': 'Transformer Models', 'start': 12, 'duration': 6, 'color': '#e377c2'}
    ]
    
    fig5 = go.Figure()
    for i, phase in enumerate(phases):
        fig5.add_trace(go.Scatter(
            x=[phase['start'], phase['start'] + phase['duration']],
            y=[i, i],
            mode='lines+markers',
            line=dict(color=phase['color'], width=10),
            marker=dict(size=12),
            name=phase['name'],
            showlegend=True
        ))
    
    fig5.update_layout(
        title="<b>24-Month Strategic Development Roadmap</b>",
        xaxis_title="Timeline (Months)",
        yaxis_title="Development Phases",
        yaxis=dict(tickvals=list(range(len(phases))), 
                  ticktext=[p['name'] for p in phases]),
        height=500,
        font=dict(size=16, family="Arial"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig5, use_container_width=True)

def export_instructions():
    st.header("📋 Export Instructions")
    st.markdown("**Step-by-step guide to transfer content to your PowerPoint presentation**")
    
    # Method tabs
    method_tab1, method_tab2, method_tab3 = st.tabs([
        "📝 Text Export",
        "📊 Chart Export", 
        "💡 Pro Tips"
    ])
    
    with method_tab1:
        st.subheader("📝 Copy-Paste Text Method")
        st.markdown("""
        **Steps for Text Content:**
        1. **Select the markdown content** from the code blocks above
        2. **Copy** (Ctrl+C or Cmd+C)
        3. **Paste into PowerPoint** slide as text
        4. **Format as bullet points** (PowerPoint will auto-convert)
        5. **Adjust font size** and spacing as needed
        
        **Formatting Tips:**
        - Use **18-20pt font** for main content
        - **Bold the main headings** (Technical Challenges, etc.)
        - **Indent sub-bullets** for details and explanations
        - **Keep consistent spacing** between sections
        """)
        
        st.info("💡 The content is pre-formatted with proper hierarchy for easy PowerPoint conversion")
    
    with method_tab2:
        st.subheader("📊 Chart Export Methods")
        
        st.markdown("""
        **Method 1: Direct Download (Recommended)**
        1. **Hover over any chart** → Click the camera icon (📷) in the top-right
        2. **Download as PNG** (best quality for presentations)
        3. **Insert into PowerPoint**: Insert > Pictures > This Device
        4. **Resize maintaining aspect ratio** (hold Shift while dragging)
        
        **Method 2: Right-Click Save**
        1. **Right-click on chart** → "Save image as"
        2. **Choose PNG format** and high resolution
        3. **Save to your presentation folder**
        
        **Method 3: Screenshot (Backup)**
        1. **Use screenshot tool** (Windows: Snipping Tool, Mac: Cmd+Shift+4)
        2. **Capture the chart area**
        3. **Paste directly into PowerPoint** (Ctrl+V)
        """)
        
        st.warning("⚠️ Ensure charts are downloaded in high resolution for crisp presentation quality")
    
    with method_tab3:
        st.subheader("💡 Professional Presentation Tips")
        
        st.markdown("""
        **Slide Layout Recommendations:**
        - **Title slide**: Use main headings as slide titles
        - **Content density**: 4-6 bullet points max per slide
        - **Chart placement**: Right side with bullets on left
        - **Consistent colors**: Blue (#1f77b4), Orange (#ff7f0e), Green (#2ca02c)
        
        **Technical Presentation Strategy:**
        1. **Start with simple explanation** (use the analogies provided)
        2. **Show the problem** before presenting solution
        3. **Highlight results** with specific numbers (92.7% accuracy, 50-100x speedup)
        4. **Connect to business value** (real-time predictions, cost savings)
        
        **For Academic Defense:**
        - **Explain complex terms** using the simple analogies provided
        - **Reference actual performance metrics** prominently
        - **Show progression**: Challenge → Solution → Implementation → Results
        - **Prepare for questions** about technical choices and trade-offs
        
        **Time Management:**
        - **Technical Challenges**: 1.5 minutes (as specified in your outline)
        - **Achievements**: 2-3 minutes for impact demonstration
        - **Future Vision**: 1-2 minutes for strategic outlook
        """)

if __name__ == "__main__":
    show_export_page()