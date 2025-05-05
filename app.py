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

# Set page configuration
st.set_page_config(
    page_title="Virtual Judge - Delhi High Court",
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

# Title and description
#st.title("Delhi High Court Virtual Judge")
#st.markdown("""
#This application uses a fine-tuned LegalBERT model to analyze legal documents, find similar cases from the 
#Delhi High Court database, and predict potential judgments based on precedent.
#""")

# Sidebar for controls and system information
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

# Import other dependencies after displaying system information
from utils.pdf_processor import extract_text_from_pdf
from utils.vector_store import VectorStore
from utils.judgment_predictor import predict_judgment
from utils.visualization import plot_case_similarity

# background color 
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #1d1a39, #e8bcb9);
            color: black;
        }
    </style>
""", unsafe_allow_html=True)


# Main content area with tabs
tabs = st.tabs(["Home","About","Upload Document", "Similar Cases", "Judgment Prediction"])#, "Search Cases", "Training"])

with tabs[0]:
   html_code = """
<div style="
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(to right, rgba(123, 187, 255, 0.8), rgba(255, 255, 255, 0.8));
  padding: 60px 80px;
  border-radius: 40px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
  font-family: 'Segoe UI', sans-serif;
  backdrop-filter: blur(10px);
">

  <div style="flex: 1; padding-right: 50px;">
    <h1 style="
      color: #050f2a;
      font-size: 60px;
      margin: 0 0 20px;
      line-height: 1.2;
      text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    ">
      Welcome to the <br> Virtual Court
    </h1>
    <p style="
      color: #333;
      font-size: 20px;
      max-width: 500px;
    ">
      Experience the future of justice from the comfort of your screen.
    </p>
  </div>

  <div style="flex: 1; text-align: right;">
    <img src="https://ichef.bbci.co.uk/ace/standard/976/cpsprodpb/14AD5/production/_95739648_gettyimages-487787078.jpg"
         alt="Courtroom Image"
         style="
           max-width: 100%;
           height: auto;
           border-radius: 16px;
           box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
           transition: transform 0.3s ease;
         "
         onmouseover="this.style.transform='scale(1.03)'"
         onmouseout="this.style.transform='scale(1)'"
    />
  </div>

</div>
"""

st.html(html_code)

    #st.code(html_code,language="html")

with tabs[1]:
   with tabs[1]:
    st.markdown("""
        <style>
            .about-container {
                background: linear-gradient(135deg, #1b263b 0%, #293b5f 100%);
                border-radius: 30px;
                padding: 50px 40px;
                color: #f2fdff;
                box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
                font-family: 'Segoe UI', sans-serif;
                max-width: 1000px;
                margin: auto;
                animation: fadeIn 1.5s ease-in-out;
            }

            .about-title {
                font-size: 60px;
                color: #7bbbff;
                text-align: center;
                margin-bottom: 30px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
            }

            .about-text {
                font-size: 18px;
                line-height: 1.75;
                text-align: justify;
                color: #e0f7ff;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 768px) {
                .about-title {
                    font-size: 38px;
                }
                .about-text {
                    font-size: 16px;
                }
            }
        </style>

        <div class="about-container">
            <div class="about-title">ABOUT US</div>
            <div class="about-text">
                <strong>Virtual Court</strong> is an AI-powered legal assistance platform that leverages a fine-tuned LegalBERT model to streamline judicial processes and support legal analysis.
                <br><br>
                This application is designed to analyze uploaded legal documents—such as case descriptions, petitions, or judgments—and extract key legal semantics. It then searches the Delhi High Court case database to identify the most relevant precedents based on contextual and legal similarity.
                <br><br>
                By comparing the current case with historical judgments, the system predicts possible outcomes, offering data-driven insights to lawyers, litigants, or researchers.
                <br><br>
                The integration of Natural Language Processing (NLP) with legal domain knowledge enables Virtual Court to provide meaningful case recommendations, enhance legal research efficiency, and simulate judgment reasoning in a virtual environment.
                <br><br>
                This system represents a significant step toward digital transformation in the judiciary by making legal intelligence more accessible, transparent, and scalable.
            </div>
        </div>
    """, unsafe_allow_html=True)


with tabs[2]:
    st.markdown("""
<style>
.custom-header {
    font-size: 45px;
    font-weight: bold;
    color: #050f2a;
    text-align: center;
    padding: 10px;
    margin-bottom: 10px;
}
.subtext {
    font-size: 20px;
    color: #7f8c8d;
    text-align: center;
    margin-top: -10px;
}
</style>

<div class="custom-header">Upload Legal Document</div>
""", unsafe_allow_html=True)

# Custom CSS & HTML for PDF Upload
upload_html = """
<style>
.upload-card {
    background-color: rgba(123, 187, 255, 0.8);
    border: #c3c3c3;
    padding: 30px;
    border-radius: 12px;
    text-align: center;
    transition: 0.3s;
}
.upload-card:hover {
    border-color: #e8bcb9;
    background-color: rgba(255, 255, 255, 0.8);
}
.upload-icon {
    font-size: 100px;
    color: #4a90e2;
}
.upload-text {
    font-size: 18px;
    color: #333;
    margin-top: 10px;
}
</style>

<div class="upload-card">
    <div class="upload-icon">📄</div>
    <div class="upload-text"><strong>Upload your Legal Document (PDF)</strong></div>
    <p style="color: #666;">Only PDF files are supported. Max size: 20MB</p>
</div>
"""

# Display custom upload UI
st.markdown(upload_html, unsafe_allow_html=True)

# Streamlit uploader (functional, underneath styled block)
uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    st.success("✅ PDF uploaded successfully!")
    # (Optional) You can process the file here

    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
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
                            
                            # Add enhanced analysis to the judgment prediction
                            if enhanced_analysis.get('enhanced', False):
                                st.session_state.judgment_prediction['enhanced_analysis'] = enhanced_analysis
                                st.success("Enhanced legal analysis with GPT-4o generated successfully!")
                        except Exception as e:
                            st.warning(f"OpenAI enhancement failed: {str(e)}")
                
                st.success("Document processed successfully! Navigate to the 'Similar Cases' and 'Judgment Prediction' tabs to see results.")
            
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
    else:
        st.info("Please upload a PDF document to begin analysis.")

with tabs[3]:
    st.markdown("""
<style>
.custom-header {
    font-size: 45px;
    font-weight: bold;
    color: #050f2a;
    text-align: center;
    padding: 10px;
    margin-bottom: 10px;
}
.subtext {
    font-size: 20px;
    color: #7f8c8d;
    text-align: center;
    margin-top: -10px;
}
</style>

<div class="custom-header">Similar Cases</div>
""", unsafe_allow_html=True)

    
    if st.session_state.similar_cases is not None:
        st.subheader("Top Similar Cases from Delhi High Court")
        
        for i, case in enumerate(st.session_state.similar_cases):
            with st.expander(f"Case #{i+1}: {case['title']} ({case['similarity_score']:.2f} similarity)"):
                st.markdown(f"**Case Number:** {case['case_number']}")
                st.markdown(f"**Date:** {case['date']}")
                st.markdown(f"**Judges:** {case['judges']}")
                st.markdown("**Summary:**")
                st.markdown(case['summary'])
                st.markdown("**Key Points:**")
                for point in case['key_points']:
                    st.markdown(f"- {point}")
        
        # Visualization of case similarity
        st.subheader("Case Similarity Visualization")
        fig = plot_case_similarity(st.session_state.similar_cases)
        st.pyplot(fig)
    else:
        st.info("No document has been processed yet. Please upload a document in the 'Upload Document' tab.")

with tabs[4]:
    st.markdown("""
<style>
.custom-header {
    font-size: 45px;
    font-weight: bold;
    color: #050f2a;
    text-align: center;
    padding: 10px;
    margin-bottom: 10px;
}
.subtext {
    font-size: 20px;
    color: #7f8c8d;
    text-align: center;
    margin-top: -10px;
}
</style>

<div class="custom-header">Judgement Prediction</div>
""", unsafe_allow_html=True)

    if st.session_state.judgment_prediction is not None:
        st.subheader("Predicted Judgment")
        
        # Create tabs for different views of the judgment
        tab_options = ["Summary", "Detailed Analysis", "Professional Judgment", "Legal Principles", "Case Parties & References"]
        
        # Add Enhanced Analysis tab if OpenAI analysis is available
        if 'enhanced_analysis' in st.session_state.judgment_prediction:
            tab_options.append("AI-Enhanced Analysis")
            
        judgment_tabs = st.tabs(tab_options)
        
        with judgment_tabs[0]:
            # Display confidence score
            st.markdown(f"**Confidence Score:** {st.session_state.judgment_prediction['confidence']:.2f}")
            
            # Display prediction category (e.g., Allowed, Dismissed, etc.)
            st.markdown(f"**Prediction:** {st.session_state.judgment_prediction['prediction']}")
            
            # Display parties if available
            if 'parties' in st.session_state.judgment_prediction and st.session_state.judgment_prediction['parties']:
                parties = st.session_state.judgment_prediction['parties']
                st.subheader("Case Parties")
                if parties.get('petitioner'):
                    st.markdown(f"**Petitioner/Appellant:** {parties.get('petitioner')}")
                if parties.get('respondent'):
                    st.markdown(f"**Respondent/Defendant:** {parties.get('respondent')}")
            
            # Get case type from judgment prediction if available
            case_type = st.session_state.judgment_prediction.get('case_type', 'General Case')
            
            # Display case type context with OpenAI if available
            if OPENAI_AVAILABLE:
                try:
                    with st.expander("Legal context for this case type", expanded=True):
                        case_context = get_legal_context(case_type, "Delhi High Court")
                        st.markdown(case_context)
                except Exception as e:
                    st.warning(f"Could not fetch case type context: {str(e)}")
            
            # Display reasoning
            st.subheader("Reasoning")
            st.markdown(st.session_state.judgment_prediction['reasoning'])
            
            # Display relevant precedents
            st.subheader("Relevant Precedents")
            for precedent in st.session_state.judgment_prediction['precedents']:
                st.markdown(f"- **{precedent['case']}**: {precedent['relevance']}")
        
        with judgment_tabs[1]:
            # Display liability determination
            if 'liability_determination' in st.session_state.judgment_prediction:
                liability = st.session_state.judgment_prediction['liability_determination']
                
                st.subheader("Liability Determination")
                
                if liability.get('primary_liability'):
                    st.markdown(f"**Primary Liability:** {liability.get('primary_liability')}")
                    
                if liability.get('secondary_liability'):
                    st.markdown(f"**Secondary Liability:** {liability.get('secondary_liability')}")
                    
                if liability.get('liability_ratio'):
                    st.markdown(f"**Liability Ratio:** {liability.get('liability_ratio')}")
                    
                st.markdown(f"**Petitioner Claims Established:** {liability.get('petitioner_claims_established', 'Unknown')}")
                st.markdown(f"**Respondent Defense Valid:** {liability.get('respondent_defense_valid', 'Unknown')}")
                
                # Display key findings
                if 'key_findings' in liability and liability['key_findings']:
                    st.subheader("Key Findings of Fact")
                    for i, finding in enumerate(liability['key_findings'], 1):
                        st.markdown(f"{i}. {finding}")
            else:
                st.info("Detailed liability analysis not available for this document.")
                
            # Display legal remedy
            if 'legal_remedy' in st.session_state.judgment_prediction:
                st.subheader("Legal Remedy")
                st.markdown(st.session_state.judgment_prediction['legal_remedy'])
            else:
                st.info("Legal remedy recommendations not available for this document.")
                
        with judgment_tabs[2]:
            # Display professional analysis
            if 'professional_analysis' in st.session_state.judgment_prediction:
                st.markdown(st.session_state.judgment_prediction['professional_analysis'])
            else:
                st.info("Professional legal analysis not available for this document.")
                
        with judgment_tabs[3]:
            # Display legal principles applied
            st.subheader("Legal Principles Applied")
            for principle in st.session_state.judgment_prediction['legal_principles']:
                st.markdown(f"- {principle}")
        
        with judgment_tabs[4]:
            # Display detailed party information
            if 'parties' in st.session_state.judgment_prediction and st.session_state.judgment_prediction['parties']:
                parties = st.session_state.judgment_prediction['parties']
                st.subheader("Case Parties")
                st.markdown(f"**Petitioner/Appellant:** {parties.get('petitioner', 'Not identified')}")
                st.markdown(f"**Respondent/Defendant:** {parties.get('respondent', 'Not identified')}")
            else:
                st.info("Party information could not be extracted from this document.")
            
            # Display legal references
            if 'legal_references' in st.session_state.judgment_prediction and st.session_state.judgment_prediction['legal_references']:
                legal_refs = st.session_state.judgment_prediction['legal_references']
                
                st.subheader("Legal References")
                if len(legal_refs) > 0:
                    # Create a table for legal references
                    st.markdown("| Statute/Law | Section/Article | Full Reference |")
                    st.markdown("|-------------|----------------|----------------|")
                    
                    for ref in legal_refs:
                        statute = ref.get('statute', 'Unknown')
                        section = ref.get('section', 'Unknown')
                        full_ref = ref.get('full_reference', 'Unknown')
                        
                        st.markdown(f"| {statute} | {section} | {full_ref} |")
                else:
                    st.info("No specific legal statutes or sections were identified in this document.")
            else:
                st.info("Legal references could not be extracted from this document.")
        
        # Display AI-Enhanced analysis if available
        if 'enhanced_analysis' in st.session_state.judgment_prediction and len(tab_options) > 5:
            with judgment_tabs[5]:
                enhanced = st.session_state.judgment_prediction['enhanced_analysis']
                
                st.subheader("AI-Enhanced Legal Analysis 🤖")
                st.info("This analysis is powered by OpenAI's GPT-4o model, providing a professional legal perspective on the case.")
                
                # Display the comprehensive analysis
                if 'comprehensive_analysis' in enhanced:
                    st.subheader("Comprehensive Analysis")
                    st.markdown(enhanced['comprehensive_analysis'])
                
                # Display the relevant citations
                if 'relevant_citations' in enhanced and enhanced['relevant_citations']:
                    st.subheader("Relevant Citations")
                    for citation in enhanced['relevant_citations']:
                        st.markdown(f"- {citation}")
                
                # Display the recommended remedies
                if 'recommended_remedies' in enhanced and enhanced['recommended_remedies']:
                    st.subheader("Recommended Remedies")
                    for remedy in enhanced['recommended_remedies']:
                        st.markdown(f"- {remedy}")
                
                # Display key considerations
                if 'key_considerations' in enhanced and enhanced['key_considerations']:
                    st.subheader("Key Legal Considerations")
                    for consideration in enhanced['key_considerations']:
                        st.markdown(f"- {consideration}")
        
        # Disclaimer
        st.warning("""
        **Disclaimer:** This prediction is based on machine learning analysis of similar cases and should not be 
        considered legal advice. The prediction is meant to provide insight into possible outcomes based on 
        historical data, but each case is unique and may have different outcomes in court.
        """)
    else:
        st.info("No document has been processed yet. Please upload a document in the 'Upload Document' tab.")

#with tabs[5]:
#    st.header("Search Cases")
#    
#    search_query = st.text_input("Search for cases by keyword or phrase:", value=st.session_state.search_query)
#    
#    if st.button("Search") or search_query != st.session_state.search_query:
#        st.session_state.search_query = search_query
#        
#        if search_query:
#            with st.spinner("Searching cases..."):
#                # Initialize vector store and search for cases
#                vector_store = VectorStore()
#                search_results = vector_store.search_cases(search_query, top_k=10)
#                
#                if search_results:
#                    st.subheader(f"Found {len(search_results)} relevant cases")
#                    
#                    for i, case in enumerate(search_results):
#                        with st.expander(f"Case #{i+1}: {case['title']} (Relevance: {case['relevance_score']:.2f})"):
#                            st.markdown(f"**Case Number:** {case['case_number']}")
#                            st.markdown(f"**Date:** {case['date']}")
#                            st.markdown(f"**Judges:** {case['judges']}")
#                            st.markdown("**Summary:**")
#                            st.markdown(case['summary'])
#                else:
#                    st.info("No matching cases found. Try different keywords.")
#        else:
#            st.info("Enter keywords to search for relevant cases.")

#with tabs[5]:
#    st.header("Model Training")
#    
#    # Check if model is already trained
#    model_path = "./fine_tuned_model/best_model"
#    model_config_path = "./config/model_config.txt"
#    
#    if os.path.exists(model_path) and os.path.exists(model_config_path):
#        st.success("A trained model is already available!")
#        
#        # Read and display model config
#        with open(model_config_path, "r") as f:
#            config_lines = f.readlines()
#        
#        config = {}
#        for line in config_lines:
#            if "=" in line:
#                key, value = line.strip().split("=", 1)
#                config[key] = value
#        
#        st.markdown(f"**Model Type:** {config.get('model_type', 'Unknown')}")
#        st.markdown(f"**Training Date:** {config.get('trained_date', 'Unknown')}")
#        
#        if st.button("Retrain Model"):
#            st.warning("This will overwrite the existing model. The process may take several minutes depending on your system.")
#            st.info("Starting training process...")
#            
#            try:
#                import subprocess
#                process = subprocess.Popen([sys.executable, "run_training.py"], 
#                                        stdout=subprocess.PIPE, 
#                                        stderr=subprocess.STDOUT,
#                                        text=True)
#                
#                # Display output in real-time
#                output_area = st.empty()
#                output_text = ""
#                
#                while True:
#                    output = process.stdout.readline()
#                    if output == '' and process.poll() is not None:
#                        break
#                    if output:
#                        output_text += output
#                        output_area.code(output_text)
#                
#                if process.returncode == 0:
#                    st.success("Training completed successfully! The application will use the new model for predictions.")
#                    st.info("Please restart the application to use the new model.")
#                else:
#                    st.error("Training failed. Please check the log for details.")
#            
#            except Exception as e:
#                st.error(f"Error during training: {str(e)}")
#    else:
#        st.info("No trained model found. You can train a model to improve prediction accuracy.")
#        
#        col1, col2 = st.columns(2)
#        
#        with col1:
#            st.markdown("""
#            **Training Process:**
#            1. The system will download a sample of Delhi High Court judgments
#            2. Prepare and clean the data for model training
#            3. Train a model (this may take 10-15 minutes)
#            4. Integrate the trained model with the application
#            """)
#        
#        with col2:
#            st.markdown("""
#            **System Requirements:**
#            - At least 4GB of available RAM
#            - Internet connection to download judgments
#            - Approximately 500MB of disk space
#            """)
#        
#        if st.button("Start Training"):
#            st.info("Starting training process...")
#            
#            try:
#                import subprocess
#                process = subprocess.Popen([sys.executable, "run_training.py"], 
#                                        stdout=subprocess.PIPE, 
#                                        stderr=subprocess.STDOUT,
#                                        text=True)
#                
#                # Display output in real-time
#                output_area = st.empty()
#                output_text = ""
#                
#                while True:
#                    output = process.stdout.readline()
#                    if output == '' and process.poll() is not None:
#                        break
#                    if output:
#                        output_text += output
#                        output_area.code(output_text)
#                
#                if process.returncode == 0:
#                    st.success("Training completed successfully! The application will use the new model for predictions.")
#                    st.info("Please restart the application to use the new model.")
#                else:
#                    st.error("Training failed. Please check the log for details.")
#            
#            except Exception as e:
#                st.error(f"Error during training: {str(e)}")
#                import traceback
#                st.code(traceback.format_exc())

# Footer information
#st.markdown("---")
#st.markdown("""
#**About this application:**  
#This Virtual Judge uses a fine-tuned LegalBERT model trained on Delhi High Court judgments to analyze legal documents
#and provide insights based on similar historical cases. The system leverages modern NLP techniques, vector embeddings,
#and similarity search to find relevant precedents.
#""")

# Add OpenAI attribution if it's available
#if OPENAI_AVAILABLE:
#    st.markdown("""
#    **Enhanced Analysis:**  
#    Professional legal analysis is enhanced using OpenAI's GPT-4o model, providing deeper legal context, 
#    detailed citations, and comprehensive legal reasoning based on the document content and case patterns.
#    """)
#else:
#    st.markdown("""
#    **Want Enhanced Analysis?**  
#    Add your OpenAI API key in the sidebar to enable GPT-4o powered legal analysis with deeper legal context,
#    relevant citations, and comprehensive professional reasoning.
#    """)
#