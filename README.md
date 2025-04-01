# Delhi High Court Virtual Judge

![Delhi High Court Virtual Judge](generated-icon.png)

An advanced AI-powered virtual judge system that leverages fine-tuned LegalBERT for sophisticated legal document analysis and precedent matching in the Delhi High Court context.

## Project Overview

The Delhi High Court Virtual Judge is an innovative application of artificial intelligence in the legal domain, designed to assist legal professionals, researchers, and litigants in analyzing case patterns and predicting potential judgment outcomes. Built on state-of-the-art natural language processing technologies, this system analyzes legal documents, identifies similar precedent cases, and generates professional legal analysis with predicted outcomes.

## Key Features

- **Document Analysis**: Upload and process legal PDF documents
- **Similar Case Identification**: Find relevant precedent cases using vector similarity search
- **Judgment Prediction**: Predict potential judgment outcomes with confidence scoring
- **Legal Analysis Generation**: Generate comprehensive legal reasoning, liability determination, and recommended remedies
- **Professional Legal Assessment**: Provides detailed legal analysis with key findings and principles
- **Case Party Information**: Accurately extracts and displays petitioner and respondent information
- **Optional GPT-4o Enhancement**: When API key is provided, generates enhanced legal analysis

## System Architecture

The system employs a modular architecture with the following components:

1. **Document Processing Module**: Extracts text from PDF documents and preprocesses it
2. **Vector Embedding System**: Converts legal documents into high-dimensional vector representations
3. **Vector Store and Similarity Search**: Stores document embeddings and enables efficient similarity search
4. **Judgment Prediction Engine**: Analyzes document content and similar cases to predict outcomes
5. **Legal Analysis Generator**: Creates comprehensive legal analysis with reasoning
6. **Streamlit User Interface**: Provides an intuitive web interface for interaction

## Technical Implementation

The system is built using Python with the following key technologies:

- **LegalBERT**: A domain-specific version of BERT fine-tuned on legal texts
- **FAISS**: For efficient similarity search of high-dimensional vectors
- **PyPDF2 and PDFPlumber**: For PDF document processing
- **Streamlit**: For creating the web-based user interface
- **Pandas and NumPy**: For data manipulation and numerical operations
- **OpenAI GPT-4o (Optional)**: For enhanced legal analysis and reasoning

<p align="center">
  
</p>

The system includes graceful degradation mechanisms to ensure functionality even when certain dependencies are unavailable, providing consistent user experience across different environments.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd delhi-high-court-virtual-judge
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

### Optional: OpenAI Integration

To enable enhanced legal analysis with GPT-4o:

1. Obtain an OpenAI API key
2. Set the OPENAI_API_KEY environment variable:
   ```
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage Guide

1. **Upload Document**: Use the "Upload Document" tab to submit a legal document (PDF format)
2. **View Similar Cases**: Navigate to the "Similar Cases" tab to see precedent cases with similarity scores
3. **Judgment Prediction**: View the predicted judgment and confidence score in the "Judgment Prediction" tab
4. **Explore Legal Analysis**: Review the detailed legal analysis, including:
   - Case summary and reasoning
   - Liability determination
   - Legal principles applied
   - Recommended legal remedies
5. **Search Cases**: Use the "Search Cases" tab to find specific cases in the database
6. **Model Training**: Access model training and enhancement options in the "Training" tab


## System Capabilities

- Processes legal documents up to hundreds of pages
- Identifies similar cases with high precision using vector similarity
- Predicts judgment outcomes with confidence scores typically ranging from 0.80 to 0.95
- Generates professional legal analysis with liability determinations and remedies
- Extracts and displays party information (petitioner/respondent) accurately
- Adapts to different document formats and structures

## Limitations

- Predictions are based on historical patterns and may not reflect recent legal precedent shifts
- Performance depends on representation of similar case types in the training data
- Not a replacement for legal professionals; designed as a supplementary analytical tool
- Complete functionality requires all dependencies; operates in degraded mode when some are unavailable

## Project Structure

```
delhi-high-court-virtual-judge/
├── app.py                  # Main Streamlit application
├── data_downloader.py      # Tools for acquiring judgment data
├── train_model.py          # Model training functionality
├── convert_to_docx.py      # Document conversion utilities
├── run_training.py         # Training pipeline orchestration
├── Project_Synopsis.html   # Project documentation
├── data/                   # Data storage
│   ├── raw/                # Raw judgment documents
│   └── processed/          # Processed judgment data
├── model/                  # Model definitions
│   ├── embedding_utils.py  # Document embedding utilities
│   ├── model_trainer.py    # Model training implementation
│   └── use_trained_model.py # Model inference code
└── utils/                  # Utility functions
    ├── data_loader.py      # Data loading utilities
    ├── judgment_predictor.py # Judgment prediction logic
    ├── openai_integration.py # Optional GPT-4o integration
    └── vector_store.py     # Vector storage implementation
```

## Acknowledgements

- Delhi High Court public resources for providing judgment data
- LegalBERT creators for the pre-trained model foundation
- The Streamlit team for the interactive web framework

## License

MIT

---

*This project is for educational and research purposes only. The predictions and analysis generated should not be considered legal advice.*
