import streamlit as st
import base64
from pathlib import Path

def show_about_author():
    """Display comprehensive About Author page highlighting career transition and achievements"""
    
    st.set_page_config(
        page_title="About Author - Jevin Tan",
        page_icon="👨‍💻",
        layout="wide"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .skill-badge {
        background: #e8f4f8;
        color: #1f77b4;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .timeline-item {
        border-left: 3px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .achievement-card {
        background: var(--background-color);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--secondary-background-color);
        margin: 1rem 0;
        color: var(--text-color);
    }
    .contact-info {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>👨‍💻 About the Author</h1>
        <h2>Tan Jue Hao Jevin</h2>
        <h3>From Pharmacist to AI/ML Engineer</h3>
        <p><em>Transforming Healthcare Knowledge into AI Innovation</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Professional Photo
        st.markdown("### 📸 Professional Profile")
        try:
            # Display the professional photo
            st.image("attached_assets/IMG-20240905-WA0001_1751505250491.jpg", 
                    caption="Jevin Tan - AI/ML Developer", 
                    width=300)
        except:
            st.info("Professional photo available in project files")
        
        # Contact Information
        st.markdown("""
        <div class="contact-info">
            <h4>📞 Contact Information</h4>
            <p><strong>📧 Email:</strong> jevintanjh@gmail.com</p>
            <p><strong>📱 Phone:</strong> +65 96347825</p>
            <p><strong>🔗 LinkedIn:</strong> linkedin.com/in/jevin-tan-consult/</p>
            <p><strong>📍 Location:</strong> Singapore</p>
            <p><strong>🌏 Languages:</strong> English (Highly Proficient), Chinese/Mandarin (Proficient), Bahasa Indonesia (Basic), Burmese (Conversational)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Career Transformation Story
        st.markdown("""
        <div class="section-header">
            <h3>🚀 Career Transformation Journey</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **From Healthcare Expert to AI Innovation Leader**
        
        With 14+ years of extensive medical and pharmaceutical industry experience, I've embarked on an exciting career transformation into AI/ML development. This unique journey combines deep healthcare domain knowledge with cutting-edge artificial intelligence capabilities, positioning me to drive meaningful innovation in healthcare technology.
        
        **Why the Transition?**
        - **Passion for Technology**: Long-standing interest in leveraging technology to solve complex healthcare challenges
        - **Industry Insight**: Deep understanding of healthcare pain points and regulatory requirements
        - **Innovation Drive**: Desire to scale impact beyond individual patient interactions to system-wide improvements
        - **Future Vision**: Commitment to transforming healthcare through AI-powered solutions
        """)
        
        # Educational Achievement
        st.markdown("""
        <div class="section-header">
            <h3>🎓 Recent Educational Achievement</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("""
        **SkillsFuture Career Transition Programme (SCTP) - Associate AI/ML Developer**
        📅 **Completion:** July 2025 | 🏫 **Institution:** NTUC Learning Hub
        
        **Comprehensive AI/ML Curriculum:**
        - Advanced Machine Learning Models (N-BEATSx, Interpretable N-BEATS, Fast Directional Forecaster)
        - Python Programming & Data Science Stack (Pandas, NumPy, Scikit-learn)
        - Web Application Development (Streamlit, API Integration)
        - Cloud AI Platforms (Microsoft Azure ML, AutoML, Cognitive Services)
        - Model Deployment & Production Monitoring
        - Performance Validation (Walk-Forward Validation, Bootstrap Testing)
        """)

    # Core Competencies Section
    st.markdown("""
    <div class="section-header">
        <h3>💡 Core Competencies & Technical Skills</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Skills in organized categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤖 AI/Machine Learning")
        skills_ai = [
            "Python (Pandas, NumPy, Scikit-learn)",
            "Time Series Forecasting",
            "N-BEATSx Implementation", 
            "Interpretable N-BEATS",
            "Fast Directional Forecaster",
            "TensorFlow/PyTorch",
            "Streamlit Development",
            "Plotly Visualization"
        ]
        for skill in skills_ai:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ☁️ Cloud & Data Platforms")
        skills_cloud = [
            "Microsoft Azure ML",
            "Azure AutoML",
            "Azure ML Designer",
            "Azure Data Services",
            "Azure Functions",
            "Azure Container Instances",
            "Azure Cognitive Services",
            "API Integration",
            "Web Scraping"
        ]
        for skill in skills_cloud:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 🏥 Healthcare & Business")
        skills_domain = [
            "Regulatory Compliance (HSA)",
            "GMP Standards",
            "Import/Wholesale Licensing",
            "Audit Management",
            "SOP Development",
            "Market Analysis",
            "Strategic Partnerships",
            "Project Management (ASANA)",
            "Stakeholder Management"
        ]
        for skill in skills_domain:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)


    


    # Unique Value Proposition
    st.markdown("""
    <div class="section-header">
        <h3>🌟 Unique Value Proposition</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="achievement-card">
            <h4>🏥 Healthcare Domain Expertise</h4>
            <ul>
                <li><strong>14+ Years</strong> pharmaceutical industry experience</li>
                <li><strong>Regulatory Mastery:</strong> HSA, GMP, licensing frameworks</li>
                <li><strong>Business Acumen:</strong> P&L management, market expansion</li>
                <li><strong>Regional Network:</strong> Singapore, Malaysia, Myanmar, Thailand, Indonesia</li>
                <li><strong>Cross-Cultural:</strong> Multi-language capabilities across ASEAN</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="achievement-card">
            <h4>🤖 AI/ML Technical Excellence</h4>
            <ul>
                <li><strong>Advanced Modeling:</strong> Time series forecasting expertise</li>
                <li><strong>Cloud Proficiency:</strong> Microsoft Azure ecosystem</li>
                <li><strong>Full-Stack Development:</strong> From data to deployment</li>
                <li><strong>Education Focus:</strong> Knowledge transfer and mentorship</li>
                <li><strong>Industry Application:</strong> Healthcare AI optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Future Vision & Goals
    st.markdown("""
    <div class="section-header">
        <h3>🚀 Future Vision & Career Goals</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **Mission:** To leverage deep healthcare industry knowledge and comprehensive AI/ML skillset to train, empower, and scale AI capabilities across the ASEAN region, with specific focus on:
    
    🎯 **Supply Chain Optimization** - Applying AI to pharmaceutical logistics and inventory management
    
    📋 **Regulatory Compliance Automation** - Streamlining healthcare regulatory processes through intelligent systems
    
    🌏 **ASEAN Healthcare Innovation** - Building AI solutions tailored for Southeast Asian healthcare markets
    
    📚 **Technology Education** - Training and mentoring the next generation of healthcare AI professionals
    
    🔬 **Research & Development** - Contributing to cutting-edge healthcare AI research and practical applications
    """)
    
    # Call to Action
    st.markdown("""
    <div class="section-header">
        <h3>🤝 Let's Connect & Collaborate</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("""
    **Ready for Immediate Contribution**
    
    Whether you're looking for healthcare AI expertise, regulatory compliance guidance, or innovative ML solutions, I bring a unique combination of deep domain knowledge and cutting-edge technical skills.
    
    📧 **Email:** jevintanjh@gmail.com  
    📱 **Phone:** +65 96347825  
    🔗 **LinkedIn:** linkedin.com/in/jevin-tan-consult/
    
    *Available for permanent, part-time, or consulting opportunities across the ASEAN region.*
    """)

if __name__ == "__main__":
    show_about_author()