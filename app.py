import streamlit as st
import os
import tempfile
import sys

# Check dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check for OpenAI availability
try:
    from utils.openai_integration import initialize_openai, is_openai_available, enhance_legal_analysis, get_legal_context, generate_summary
    OPENAI_AVAILABLE = initialize_openai()
except ImportError:
    OPENAI_AVAILABLE = False

# Import other dependencies
from utils.pdf_processor import extract_text_from_pdf
from utils.vector_store import VectorStore
from utils.judgment_predictor import predict_judgment
from utils.visualization import plot_case_similarity

# Set page configuration
st.set_page_config(
    page_title="Delhi HC Virtual Judge",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables if they don't exist
if 'processed_document' not in st.session_state:
    st.session_state.processed_document = None
if 'similar_cases' not in st.session_state:
    st.session_state.similar_cases = None
if 'judgment_prediction' not in st.session_state:
    st.session_state.judgment_prediction = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'dependency_check' not in st.session_state:
    st.session_state.dependency_check = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"

# Apply custom theme with CSS - Enhanced based on requirements
st.markdown("""
<style>
    /* Import premium fonts - Poppins and Inter for professional look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Animated Background Canvas for particles effect */
    #particles-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.3;
    }
    
    /* Base styles for dark mode app */
    .stApp {
        background-color: #0e141e;
        background-image: radial-gradient(circle at 10% 20%, #141e30 0%, #0e1420 90%);
        font-family: 'Poppins', 'Inter', sans-serif;
        color: #e6e7eb;
        overflow-x: hidden;
    }
    
    /* Typography enhancements */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    p, div, span, .stMarkdown p {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        font-size: 16px;
    }
    
    /* Gold accent color - primary highlight */
    .gold-accent {
        color: #FFD700 !important;
    }
    
    /* Premium button styling with enhanced hover effects */
    .stButton button {
        background: linear-gradient(135deg, #FFD700 0%, #e6b800 100%) !important;
        color: #0e141e !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.8rem !important;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        text-transform: none !important;
        font-size: 16px !important;
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: 0.3px !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 7px 20px rgba(255, 215, 0, 0.4) !important;
        filter: brightness(1.05) !important;
    }
    
    .stButton button:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(255, 215, 0, 0.4) !important;
    }
    
    /* Secondary button (ghost) styling */
    .stButton [data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: 2px solid #FFD700 !important;
        color: #FFD700 !important;
        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.15) !important;
    }
    
    .stButton [data-testid="baseButton-secondary"]:hover {
        background-color: rgba(255, 215, 0, 0.1) !important;
        border-color: #FFD700 !important;
        color: #FFD700 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 7px 15px rgba(255, 215, 0, 0.2) !important;
    }
    
    /* Hide default elements */
    footer {display: none !important;}
    #MainMenu {visibility: hidden;}
    
    /* Premium glassmorphism container styling */
    .content-container {
        background: rgba(22, 28, 43, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 35px;
        margin-bottom: 28px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
        transition: all 0.4s ease;
    }
    
    .content-container:hover {
        box-shadow: 0 15px 60px rgba(0, 0, 0, 0.3), 0 0 30px rgba(255, 215, 0, 0.05);
        border: 1px solid rgba(255, 215, 0, 0.1);
    }
    
    /* Enhanced feature cards with premium hover effects */
    .feature-card {
        background: rgba(22, 28, 43, 0.7);
        border-radius: 16px;
        padding: 32px;
        height: 100%;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 7px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, transparent 0%, rgba(255, 215, 0, 0.03) 100%);
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3), 0 0 15px rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.15);
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    /* Icon animations for feature cards */
    .feature-icon {
        font-size: 42px;
        color: #FFD700;
        margin-bottom: 24px;
        display: inline-block;
        transition: all 0.5s ease;
    }
    
    .feature-card:hover .feature-icon {
        transform: scale(1.1) rotate(5deg);
        filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.5));
    }
    
    /* Enhanced subtitle text styling */
    .subtitle {
        color: #a0aec0 !important;
        font-size: 1.15rem;
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Premium metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(to right, #FFD700, #FFC107) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: -0.5px !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #a0aec0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Enhanced tab buttons styling */
    button[data-testid="baseButton-secondary"] {
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    button[data-testid="baseButton-secondary"]:hover {
        background-color: rgba(255, 215, 0, 0.1) !important;
        border-color: #FFD700 !important;
        color: #FFD700 !important;
        transform: translateY(-2px) !important;
    }
    
    button[data-testid="baseButton-primary"] {
        background-color: rgba(255, 215, 0, 0.15) !important;
        border-color: #FFD700 !important;
        color: #FFD700 !important;
        font-weight: 600 !important;
    }
    
    /* Animated blinking cursor */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    
    .cursor {
        display: inline-block;
        width: 3px;
        height: 38px;
        background-color: #FFD700;
        margin-left: 5px;
        animation: blink 1s infinite;
        position: relative;
        top: 8px;
        box-shadow: 0 0 8px rgba(255, 215, 0, 0.7);
    }
    
    /* Text reveal animation */
    @keyframes revealText {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .reveal-text {
        opacity: 0;
        animation: revealText 0.8s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }
    
    .reveal-delay-1 { animation-delay: 0.1s; }
    .reveal-delay-2 { animation-delay: 0.3s; }
    .reveal-delay-3 { animation-delay: 0.5s; }
    .reveal-delay-4 { animation-delay: 0.7s; }
    
    /* Typing animation for headings */
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    .typing-text {
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        animation: typing 2.5s steps(40, end);
    }
    
    /* Enhanced form elements styling */
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(22, 28, 43, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e6e7eb !important;
        padding: 14px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(255, 215, 0, 0.5) !important;
        box-shadow: 0 0 0 4px rgba(255, 215, 0, 0.15) !important;
        background-color: rgba(22, 28, 43, 0.9) !important;
    }
    
    /* Enhanced file uploader styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(22, 28, 43, 0.7) !important;
        border: 2px dashed rgba(255, 215, 0, 0.3) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(255, 215, 0, 0.5) !important;
        background-color: rgba(22, 28, 43, 0.8) !important;
    }
    
    /* Enhanced alert boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: rgba(255, 215, 0, 0.8) !important;
        background-image: linear-gradient(45deg, rgba(255, 215, 0, 0.8) 25%, transparent 25%, transparent 50%, rgba(255, 215, 0, 0.8) 50%, rgba(255, 215, 0, 0.8) 75%, transparent 75%, transparent) !important;
        background-size: 1rem 1rem !important;
        animation: progress-bar-stripes 1s linear infinite !important;
    }
    
    @keyframes progress-bar-stripes {
        0% { background-position: 1rem 0; }
        100% { background-position: 0 0; }
    }
    
    /* Rotating glow animation for highlighted elements */
    @keyframes rotatingGlow {
        0% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
        50% { box-shadow: 0 0 30px rgba(255, 215, 0, 0.5); }
        100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
    }
    
    /* Neon gavel floating animation */
    @keyframes floatEffect {
        0% { filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.6)); transform: translateY(0); }
        50% { filter: drop-shadow(0 0 20px rgba(0, 240, 255, 0.8)); transform: translateY(-7px); }
        100% { filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.6)); transform: translateY(0); }
    }
    
    .glow-effect {
        animation: rotatingGlow 3s infinite;
    }
</style>

<!-- Particles animation for background using vanilla JS -->
<canvas id="particles-canvas"></canvas>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var canvas = document.getElementById('particles-canvas');
        var ctx = canvas.getContext('2d');
        
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        var particles = [];
        var particleCount = 100;
        
        for (var i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 2 + 1,
                color: 'rgba(255, 215, 0, ' + (Math.random() * 0.15 + 0.05) + ')',
                speedX: Math.random() * 0.5 - 0.25,
                speedY: Math.random() * 0.5 - 0.25
            });
        }
        
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            for (var i = 0; i < particleCount; i++) {
                var particle = particles[i];
                
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
                ctx.fillStyle = particle.color;
                ctx.fill();
                
                // Draw connections between nearby particles
                for (var j = i + 1; j < particleCount; j++) {
                    var particle2 = particles[j];
                    var dx = particle.x - particle2.x;
                    var dy = particle.y - particle2.y;
                    var distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < 100) {
                        ctx.beginPath();
                        ctx.strokeStyle = 'rgba(255, 215, 0, ' + (0.1 - distance/1000) + ')';
                        ctx.lineWidth = 0.5;
                        ctx.moveTo(particle.x, particle.y);
                        ctx.lineTo(particle2.x, particle2.y);
                        ctx.stroke();
                    }
                }
                
                // Update particle position
                particle.x += particle.speedX;
                particle.y += particle.speedY;
                
                // Bounce off edges
                if (particle.x < 0 || particle.x > canvas.width) {
                    particle.speedX = -particle.speedX;
                }
                if (particle.y < 0 || particle.y > canvas.height) {
                    particle.speedY = -particle.speedY;
                }
            }
            
            requestAnimationFrame(draw);
        }
        
        draw();
    });
</script>
""", unsafe_allow_html=True)

# Top header row with logo and navigation
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <div style="margin-right: 15px;">
        <img src="https://i.imgur.com/YqcwkC4.png" width="42" alt="gavel icon" style="filter: drop-shadow(0 0 8px rgba(0, 240, 255, 0.7));" />
    </div>
    <div>
        <div style="font-weight: bold; color: #FFD700;">Virtual Judge</div>
        <div style="font-size: 0.8rem; color: #a0aec0;">AI Legal Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation tabs using Streamlit components
col1, col2, col3, col4, col5 = st.columns(5)

# Create tab buttons
if col1.button("Home", key="tab_home", use_container_width=True, 
              type="primary" if st.session_state.active_tab == "Home" else "secondary"):
    st.session_state.active_tab = "Home"
    st.rerun()
    
if col2.button("Dashboard", key="tab_dashboard", use_container_width=True, 
              type="primary" if st.session_state.active_tab == "Dashboard" else "secondary"):
    st.session_state.active_tab = "Dashboard"
    st.rerun()
    
if col3.button("Upload", key="tab_upload", use_container_width=True, 
              type="primary" if st.session_state.active_tab == "Upload" else "secondary"):
    st.session_state.active_tab = "Upload"
    st.rerun()
    
if col4.button("Analysis", key="tab_analysis", use_container_width=True, 
              type="primary" if st.session_state.active_tab == "Analysis" else "secondary"):
    st.session_state.active_tab = "Analysis"
    st.rerun()
    
if col5.button("Judge", key="tab_judge", use_container_width=True, 
              type="primary" if st.session_state.active_tab == "Judge" else "secondary"):
    st.session_state.active_tab = "Judge"
    st.rerun()

# Divider
st.markdown("<hr style='border: none; height: 1px; background-color: rgba(255, 255, 255, 0.1); margin: 20px 0;'>", unsafe_allow_html=True)

# Show content based on active tab
if st.session_state.active_tab == "Home":
    # Hero Section - Home Tab
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown("""
        <div class="content-container glass-effect glow-effect" style="padding: 45px;">
            <div style="margin-bottom: 28px;">
                <h1 style="font-size: 3.8rem; font-weight: 700; line-height: 1.2; letter-spacing: -1px;">
                    <span class="typing-text reveal-text reveal-delay-1" style="color: white;">AI-Powered</span><br>
                    <span class="reveal-text reveal-delay-2" style="color: #FFD700;">|Virtual Judge|</span>
                    <span class="cursor"></span>
                </h1>
            </div>
            
            <p class="subtitle reveal-text reveal-delay-3" style="margin-bottom: 35px; font-size: 1.15rem; line-height: 1.6; color: #a0aec0;">
                Experience the future of legal analysis with our AI-driven legal assistance platform. Upload your case documents and receive instant insights and predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload Case Files", use_container_width=True):
                st.session_state.active_tab = "Upload"
                st.rerun()
                
        with col2:
            if st.button("Consult Virtual Judge", use_container_width=True, type="secondary"):
                st.session_state.active_tab = "Judge"
                st.rerun()
    
    with right_col:
        # Enhanced logo with animation effects
        st.markdown("""
        <div class="content-container glass-effect" style="text-align: center; padding: 35px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div class="reveal-text reveal-delay-1" style="font-size: 75px; margin-bottom: 20px; animation: floatEffect 3s infinite;">
                <img src="https://i.imgur.com/YqcwkC4.png" width="160" alt="gavel icon" style="filter: drop-shadow(0 0 15px rgba(0, 240, 255, 0.7)); transition: all 0.5s ease;" />
            </div>
            <h3 class="reveal-text reveal-delay-2" style="color: #FFD700; margin-bottom: 10px; font-weight: 600; font-size: 1.8rem;">Virtual Judge</h3>
            <p class="reveal-text reveal-delay-3" style="color: #a0aec0; font-weight: 300;">AI-Powered Legal Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced feature cards with premium animations and styling
    st.markdown("<div class='reveal-text reveal-delay-4'><br></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card reveal-text reveal-delay-4" style="text-align: center; height: 230px;">
            <div class="feature-icon">
                <div style="background: rgba(255, 215, 0, 0.1); border-radius: 50%; width: 90px; height: 90px; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; box-shadow: 0 10px 25px rgba(255, 215, 0, 0.2);">   
                    <img src="https://img.icons8.com/fluency/96/scales.png" width="60" alt="scales icon" style="transform: translateY(-2px);" />
                </div>
            </div>
            <h3 style="color: white; margin-bottom: 14px; font-size: 1.45rem; font-weight: 600;">Legal Analysis</h3>
            <p style="color: #a0aec0; font-size: 0.95rem; line-height: 1.5;">AI-powered analysis of legal documents and case files with intelligent reasoning</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card reveal-text reveal-delay-5" style="text-align: center; height: 230px;">
            <div class="feature-icon">
                <div style="background: rgba(255, 215, 0, 0.1); border-radius: 50%; width: 90px; height: 90px; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; box-shadow: 0 10px 25px rgba(255, 215, 0, 0.2);">   
                    <img src="https://img.icons8.com/fluency/96/search.png" width="60" alt="search icon" style="transform: translateY(-2px);" />
                </div>
            </div>
            <h3 style="color: white; margin-bottom: 14px; font-size: 1.45rem; font-weight: 600;">Case Similarity</h3>
            <p style="color: #a0aec0; font-size: 0.95rem; line-height: 1.5;">Find similar cases in our comprehensive database for precedent research</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card reveal-text reveal-delay-6" style="text-align: center; height: 230px;">
            <div class="feature-icon">
                <div style="background: rgba(255, 215, 0, 0.1); border-radius: 50%; width: 90px; height: 90px; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; box-shadow: 0 10px 25px rgba(255, 215, 0, 0.2);">   
                    <img src="https://img.icons8.com/fluency/96/bot.png" width="60" alt="robot icon" style="transform: translateY(-2px);" />
                </div>
            </div>
            <h3 style="color: white; margin-bottom: 14px; font-size: 1.45rem; font-weight: 600;">AI Predictions</h3>
            <p style="color: #a0aec0; font-size: 0.95rem; line-height: 1.5;">Get judgment predictions based on case facts and established legal precedents</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.active_tab == "Dashboard":
    st.header("Dashboard")
    st.write("View your recent case history and analysis results")
    
    # Placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cases Analyzed", "24")
    col2.metric("Success Rate", "78%")
    col3.metric("Avg. Similarity", "0.82")
    col4.metric("AI Predictions", "92% Accurate")

elif st.session_state.active_tab == "Upload":
    st.header("Upload Legal Document")
    st.write("Upload your case file to analyze and get AI-powered insights")

    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success("✅ PDF uploaded successfully!")

        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Extract text from PDF
                document_text = extract_text_from_pdf(tmp_file_path)

                # Display document preview
                st.subheader("Document Preview")
                preview_text = document_text[:1000] + "..." if len(document_text) > 1000 else document_text
                st.text_area("Document Content (Preview)", preview_text, height=250)

                # Generate document summary with OpenAI if available
                if OPENAI_AVAILABLE:
                    with st.spinner("Generating document summary with GPT-4o..."):
                        try:
                            summary = generate_summary(document_text, max_words=300)
                            st.subheader("AI-Generated Document Summary")
                            st.info(summary)
                        except Exception as e:
                            st.error(f"Error generating document summary: {str(e)}")

                # Save processed document in session state for other tabs
                st.session_state.processed_document = document_text

                # Initialize vector store and find similar cases
                vector_store = VectorStore()
                st.session_state.similar_cases = vector_store.find_similar_cases(document_text, top_k=5)

                # Generate judgment prediction
                st.session_state.judgment_prediction = predict_judgment(document_text, st.session_state.similar_cases)

                # If OpenAI is available, enhance the analysis with GPT-4o
                if OPENAI_AVAILABLE and 'judgment_prediction' in st.session_state:
                    with st.spinner("Enhancing analysis with GPT-4o..."):
                        try:
                            enhanced_analysis = enhance_legal_analysis(
                                document_text=document_text,
                                predicted_outcome=st.session_state.judgment_prediction['prediction'],
                                confidence=st.session_state.judgment_prediction['confidence'],
                                legal_principles=st.session_state.judgment_prediction['legal_principles'],
                                liability_determination=st.session_state.judgment_prediction['liability_determination']
                            )
                            
                            # Save in session state
                            st.session_state.enhanced_analysis = enhanced_analysis
                            
                        except Exception as e:
                            st.error(f"Error enhancing analysis with GPT-4o: {str(e)}")
                            st.session_state.enhanced_analysis = None

                st.success("Document processed successfully! Navigate to the 'Analysis' and 'Judge' tabs to see results.")

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

elif st.session_state.active_tab == "Analysis":
    st.header("Similar Legal Cases")
    
    if not st.session_state.similar_cases:
        st.info("Please upload a document in the 'Upload' tab to see similar cases.")
    else:
        # Display similar cases
        for i, case in enumerate(st.session_state.similar_cases):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(case['title'])
                    st.caption(f"Case ID: {case['case_id']} | Judgment Date: {case['judgment_date']}")
                    st.write(case['summary'])
                    st.caption(f"Outcome: {case['outcome']}")
                
                with col2:
                    st.metric("Similarity", f"{case['similarity']:.2f}")
        
        # Display visualization
        st.subheader("Case Similarity Analysis")
        plot_case_similarity(st.session_state.similar_cases)

elif st.session_state.active_tab == "Judge":
    st.header("AI Judgment Prediction")
    
    if not st.session_state.judgment_prediction:
        st.info("Please upload a document in the 'Upload' tab to see judgment predictions.")
    else:
        prediction = st.session_state.judgment_prediction
        
        # Summary, Legal Analysis, Confidence tabs
        pred_tab1, pred_tab2, pred_tab3 = st.tabs(["Summary", "Legal Analysis", "Confidence Assessment"])
        
        with pred_tab1:
            st.markdown(f"<h2 style='text-align: center;'>Predicted Judgment</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: #FFD700;'>{prediction['prediction']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Confidence: <strong>{prediction['confidence']*100:.1f}%</strong></p>", unsafe_allow_html=True)
            
            # Display enhanced analysis if available
            if 'enhanced_analysis' in st.session_state and st.session_state.enhanced_analysis:
                enhanced = st.session_state.enhanced_analysis
                st.subheader("Enhanced Analysis by GPT-4o")
                
                st.write("**AI-Enhanced Judgment Analysis:**")
                st.write(enhanced['outcome_analysis'])
                
                st.write("**Legal Principles Analysis:**")
                st.write(enhanced['legal_principles_analysis'])
                
                st.write("**Strategic Recommendations:**")
                st.write(enhanced['recommendations'])
        
        with pred_tab2:
            # Legal Analysis with detailed legal reasoning
            st.subheader("Applicable Legal Principles")
            for principle in prediction['legal_principles']:
                st.info(principle)
            
            st.subheader("Liability Determination")
            st.write(prediction['liability_determination'])
            
            st.subheader("Similar Precedents")
            for precedent in prediction['similar_precedents']:
                # Find matching case
                matching_case = next((case for case in st.session_state.similar_cases 
                                     if case['case_id'] == precedent['case_id']), None)
                if matching_case:
                    st.write(f"**{matching_case['title']}**")
                    st.caption(f"Case ID: {precedent['case_id']} | Relevance: {precedent['relevance']:.2f}")
        
        with pred_tab3:
            # Confidence Assessment
            st.subheader("Prediction Confidence Assessment")
            
            # Confidence visualization
            confidence = prediction['confidence']
            st.progress(confidence)
            
            st.write("**Confidence Analysis:**")
            st.write("The confidence score is calculated based on:")
            st.markdown("- Similarity with precedent cases\n- Consistency of outcomes in similar cases\n- Strength of legal principles application\n- Clarity of factual circumstances")
            
            st.write("**Interpretation:**")
            if confidence >= 0.7:
                st.success("This prediction has high confidence and is likely reliable.")
            elif confidence >= 0.5:
                st.warning("This prediction has moderate confidence, consider additional legal research.")
            else:
                st.error("This prediction has low confidence due to limited similar precedents or conflicting legal principles.")

# Sidebar for system information
st.sidebar.header("Settings")

# Check system information and dependencies
if not st.session_state.dependency_check:
    missing_libraries = []
    
    try:
        import PyPDF2
    except ImportError:
        missing_libraries.append("PyPDF2")
    
    try:
        import pdfplumber
    except ImportError:
        missing_libraries.append("pdfplumber")
        
    try:
        import faiss
    except ImportError:
        missing_libraries.append("FAISS")
        
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        missing_libraries.append("Transformers")
    
    st.session_state.dependency_check = True
    
    # Report missing dependencies in the sidebar
    if missing_libraries:
        st.sidebar.warning(f"Some optional dependencies are missing: {', '.join(missing_libraries)}")
        st.sidebar.info("The application will use fallback implementations where necessary.")
    else:
        st.sidebar.success("All dependencies are installed correctly.")

# Check if CUDA is available for GPU acceleration
if TORCH_AVAILABLE:
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"Running on: {device_info}")
else:
    st.sidebar.info("Running on: CPU (PyTorch not available)")

# Display OpenAI status
if OPENAI_AVAILABLE:
    st.sidebar.success("OpenAI API: Connected ✓")
else:
    st.sidebar.warning("OpenAI API: Not connected ✗")
    st.sidebar.info("Set the OPENAI_API_KEY environment variable for enhanced analysis capabilities.")
    
    # Add a button to input API key
    if st.sidebar.button("Add OpenAI API Key"):
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.sidebar.success("API Key set! Please refresh the page to use OpenAI features.")
            from utils.openai_integration import initialize_openai
            OPENAI_AVAILABLE = initialize_openai()