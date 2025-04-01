import json
import os
from typing import Dict, Any, List, Optional

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
from openai import OpenAI

# Initialize the OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = None

def initialize_openai():
    """Initialize the OpenAI client if API key is available"""
    global openai_client
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return True
    return False

def is_openai_available():
    """Check if OpenAI integration is available"""
    return OPENAI_API_KEY is not None and openai_client is not None

def enhance_legal_analysis(
    document_text: str, 
    predicted_outcome: str,
    confidence: float,
    legal_principles: List[str],
    liability_determination: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhance legal analysis with GPT-4o
    
    Args:
        document_text: Text of the legal document
        predicted_outcome: Predicted judgment outcome
        confidence: Confidence score of the prediction
        legal_principles: List of identified legal principles
        liability_determination: Dictionary with liability determination
        
    Returns:
        Dictionary with enhanced analysis
    """
    if not is_openai_available():
        return {
            "enhanced": False,
            "message": "OpenAI API key not available. Please set the OPENAI_API_KEY environment variable for enhanced analysis."
        }
    
    try:
        # Extract key details from liability determination
        primary_liability = liability_determination.get('primary_liability', 'Unknown')
        secondary_liability = liability_determination.get('secondary_liability', 'None')
        liability_ratio = liability_determination.get('liability_ratio', 'Unknown')
        key_findings = liability_determination.get('key_findings', [])
        
        # Create the prompt with all available information
        prompt = f"""
You are a legal expert analyzing a Delhi High Court case. Based on the following information, provide a comprehensive legal analysis:

Document Summary: {document_text[:1000]}... (document truncated)

Predicted Outcome: {predicted_outcome} (Confidence: {confidence:.2f})

Key Legal Principles:
{chr(10).join([f"- {principle}" for principle in legal_principles])}

Liability Determination:
- Primary Liability: {primary_liability}
- Secondary Liability: {secondary_liability}
- Liability Ratio: {liability_ratio}

Key Findings:
{chr(10).join([f"- {finding[:200]}..." if len(finding) > 200 else f"- {finding}" for finding in key_findings[:3]])}

Please provide:
1. A comprehensive legal analysis (approximately 300 words)
2. Specific citations to relevant statutes or precedents that would apply
3. Potential legal remedies or orders that would be appropriate
4. Key legal considerations the court would likely weigh in this matter

Format your response as a JSON object with the following structure:
{{
    "comprehensive_analysis": "detailed legal analysis here",
    "relevant_citations": ["citation 1", "citation 2", "etc"],
    "recommended_remedies": ["remedy 1", "remedy 2", "etc"],
    "key_considerations": ["consideration 1", "consideration 2", "etc"]
}}
"""
        
        # Make the API call
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal expert assistant providing professional analysis of Delhi High Court cases."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        # Extract the response content
        response_content = response.choices[0].message.content
        enhanced_analysis = json.loads(response_content)
        
        # Add enhanced flag
        enhanced_analysis["enhanced"] = True
        
        return enhanced_analysis
    
    except Exception as e:
        return {
            "enhanced": False,
            "message": f"Error enhancing legal analysis: {str(e)}"
        }

def get_legal_context(case_type: str, jurisdiction: str = "Delhi High Court") -> str:
    """
    Get contextual information about specific case types and jurisdictional details
    
    Args:
        case_type: Type of case (e.g., "Writ Petition", "Criminal Appeal")
        jurisdiction: Jurisdiction (default "Delhi High Court")
        
    Returns:
        Contextual information as string
    """
    if not is_openai_available():
        return "OpenAI API key not available. Please set the OPENAI_API_KEY environment variable for legal context."
    
    try:
        prompt = f"""
Provide a concise explanation (about 150 words) of the legal context for a {case_type} in the {jurisdiction}.
Include:
1. The purpose and scope of this type of case
2. Key procedural aspects
3. The standard of review or evaluation applied by the court
4. Important legal principles or statutes typically considered
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal expert providing concise information about Indian legal procedures and case types."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error fetching legal context: {str(e)}"

def generate_summary(document_text: str, max_words: int = 300) -> str:
    """
    Generate a concise summary of a legal document
    
    Args:
        document_text: Text of the document
        max_words: Maximum words in summary
        
    Returns:
        Summary as string
    """
    if not is_openai_available():
        # Return first part of document as fallback
        words = document_text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + "..."
        return document_text
    
    try:
        prompt = f"""
Summarize the following legal document in about {max_words} words, focusing on:
1. The key facts and legal issues
2. The parties involved and their claims
3. The procedural history if mentioned
4. The primary legal questions raised

Document:
{document_text[:4000]}... (document truncated)
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal expert who summarizes complex legal documents concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_words * 2,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        # Return first part of document as fallback
        words = document_text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + f"... (Error generating summary: {str(e)})"
        return document_text