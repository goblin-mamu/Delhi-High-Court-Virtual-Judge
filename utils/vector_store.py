import os
import numpy as np
import pickle
from typing import List, Dict, Any, Union, Tuple
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
from model.embedding_utils import LegalDocumentEmbedder

class VectorStore:
    def __init__(
        self, 
        index_path="./data/faiss_index",
        metadata_path="./data/metadata.pkl",
        model_path="nlpaueb/legal-bert-base-uncased",
        dimension=768,  # Default dimension for BERT embeddings
        create_if_missing=True
    ):
        """
        Initialize the vector store for legal document similarity search.
        
        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata pickle file
            model_path: Path to the model for embedding
            dimension: Embedding dimension
            create_if_missing: Whether to create a new index if none exists
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        
        # Initialize the document embedder
        self.embedder = LegalDocumentEmbedder(model_path=model_path)
        
        # Check if FAISS is available
        if not FAISS_AVAILABLE:
            print("FAISS library not available. Using mock vector store.")
            self.using_mock = True
            self.metadata = []
            return
            
        self.using_mock = False
        
        # Load or create FAISS index
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"Loading existing index from {index_path}")
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        elif create_if_missing:
            print(f"Creating new index at {index_path}")
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
            self.metadata = []
            
            # Create and save empty index
            faiss.write_index(self.index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        else:
            raise FileNotFoundError(f"Index not found at {index_path} and create_if_missing is False")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and other metadata
        """
        if not documents or self.using_mock:
            return
        
        # Generate embeddings for the documents
        texts = [doc['text'] for doc in documents]
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        for i, text in enumerate(texts):
            embeddings[i] = self.embedder.get_document_embedding(text)
        
        # Add embeddings to the index
        self.index.add(embeddings)
        
        # Add metadata
        for doc in documents:
            doc_metadata = {k: v for k, v in doc.items() if k != 'text'}
            self.metadata.append(doc_metadata)
        
        # Save updated index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def find_similar_cases(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar cases to the query document.
        
        Args:
            query_text: Text of the query document
            top_k: Number of similar cases to return
            
        Returns:
            List of similar cases with metadata and similarity scores
        """
        # Always use sample data in mock mode
        if self.using_mock:
            return self._get_sample_cases(query_text, top_k)
            
        # Check if index is empty
        if self.index.ntotal == 0:
            # Return sample data for development/testing
            return self._get_sample_cases(query_text, top_k)
        
        # Generate embedding for the query document
        query_embedding = self.embedder.get_document_embedding(query_text)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                case = self.metadata[idx].copy()
                case['similarity_score'] = float(scores[0][i])
                results.append(case)
        
        return results
    
    def search_cases(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for cases by keyword or phrase.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of cases matching the query
        """
        # Always use sample data in mock mode
        if self.using_mock:
            return self._get_sample_search_results(query, top_k)
            
        # Check if index is empty
        if self.index.ntotal == 0:
            # Return sample data for development/testing
            return self._get_sample_search_results(query, top_k)
        
        # Generate embedding for the query
        query_embedding = self.embedder.get_document_embedding(query)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                case = self.metadata[idx].copy()
                case['relevance_score'] = float(scores[0][i])
                results.append(case)
        
        return results
    
    def _get_sample_cases(self, query_text: str, count: int) -> List[Dict[str, Any]]:
        """
        Get sample cases for development/testing when no real index is available.
        
        Args:
            query_text: Text of the query document (used for deterministic randomization)
            count: Number of sample cases to return
            
        Returns:
            List of sample cases
        """
        import hashlib
        
        # Create deterministic samples based on query text
        text_hash = hashlib.md5(query_text.encode()).hexdigest()
        seed_value = int(text_hash, 16) % (2**32)
        np.random.seed(seed_value)
        
        # Delhi High Court case types and standard formats 
        case_types = [
            {"name": "Writ Petition (Civil)", "code": "W.P.(C)"},
            {"name": "Criminal Appeal", "code": "Crl.A."},
            {"name": "Civil Appeal", "code": "C.A."},
            {"name": "Arbitration Petition", "code": "ARB.P."},
            {"name": "Company Petition", "code": "CO.PET."},
            {"name": "Original Side", "code": "O.S."},
            {"name": "Letters Patent Appeal", "code": "LPA"},
            {"name": "Regular Second Appeal", "code": "RSA"},
            {"name": "Execution Petition", "code": "EX.P."}
        ]
        
        # Legal topics commonly handled in Delhi High Court
        case_topics = [
            "Land Acquisition", "Service Matter", "Recovery of Money", 
            "Tenancy Dispute", "Constitutional Challenge", "Contract Enforcement", 
            "Intellectual Property Infringement", "Corporate Insolvency", 
            "Matrimonial Dispute", "Environmental Compliance"
        ]
        
        # Well-known Delhi High Court judges
        judge_templates = [
            "Hon'ble Justice Rajiv Shakdher",
            "Hon'ble Justice Pratibha M. Singh",
            "Hon'ble Justice Sanjeev Narula",
            "Hon'ble Justice Vibhu Bakhru",
            "Hon'ble Justice Prathiba M. Singh",
            "Hon'ble Justice C. Hari Shankar",
            "Hon'ble Justice Navin Chawla",
            "Hon'ble Justice Rekha Palli",
            "Hon'ble Justice Subramonium Prasad",
            "Hon'ble Justice Yashwant Varma"
        ]
        
        # Generate realistic-looking sample cases
        samples = []
        for i in range(count):
            case_type = np.random.choice(case_types)
            case_topic = np.random.choice(case_topics)
            case_year = 2018 + (i % 6)  # Years between 2018-2023
            
            # Generate case number in proper Delhi HC format
            case_number = f"{10000+i}/{case_year}"
            case_id = f"{case_type['code']} {case_number}"
            
            # Generate realistic party names
            petitioners = [
                "Aditya Sharma", "Raj Enterprises Pvt. Ltd.", "Sunil Kumar", 
                "Delhi Residents Welfare Association", "Vikas Construction Co.", 
                "Kavita Gupta", "Modern School", "Nandini Singh", "Bharat Industries", 
                "Ravi Textiles Ltd.", "Indian Medical Association", "RK Builders"
            ]
            
            respondents = [
                "Union of India", "State of Delhi", "Delhi Development Authority",
                "Municipal Corporation of Delhi", "Delhi Electricity Regulatory Commission",
                "Delhi Metro Rail Corporation", "Delhi Urban Shelter Improvement Board",
                "All India Institute of Medical Sciences", "Delhi Transport Corporation",
                "Delhi Jal Board", "Delhi Police", "Income Tax Department"
            ]
            
            petitioner = petitioners[i % len(petitioners)]
            respondent = respondents[i % len(respondents)]
            
            # Clean party names - no additional descriptive text
            petitioner = petitioner.split(':')[0] if ':' in petitioner else petitioner
            respondent = respondent.split(':')[0] if ':' in respondent else respondent
            
            title = f"{petitioner} vs {respondent}"
            
            # Generate summary based on topic and case type
            summary_templates = [
                f"This {case_type['name']} involves a {case_topic.lower()} matter where {petitioner} challenged the decision of {respondent}.",
                f"A {case_topic.lower()} case where {petitioner} sought relief against the order passed by {respondent}.",
                f"This case concerns {case_topic.lower()} issues where {petitioner} claimed violation of statutory rights by {respondent}.",
                f"{petitioner} filed this {case_type['name']} alleging irregularities in a {case_topic.lower()} matter by {respondent}.",
                f"A dispute regarding {case_topic.lower()} where {respondent}'s actions were challenged as being ultra vires by {petitioner}."
            ]
            
            summary = np.random.choice(summary_templates)
            
            # Create key points that sound like legal principles
            key_point_templates = [
                "Administrative actions must adhere to principles of natural justice and fair procedure.",
                f"In matters of {case_topic.lower()}, authorities have limited discretionary powers defined by the statute.",
                "When statutory procedures are violated, the resulting decision becomes void ab initio.",
                f"The {case_type['name']} jurisdiction cannot be invoked when alternative remedies are available and efficacious.",
                "Substantial compliance with statutory requirements may be sufficient if the purpose of the law is fulfilled.",
                "The burden of proof lies with the party alleging violation of statutory provisions or fundamental rights.",
                f"In {case_topic.lower()} cases, the court must balance competing public interest and individual rights."
            ]
            
            # Select 3 unique key points
            key_point_indices = np.random.choice(len(key_point_templates), 3, replace=False)
            key_points = [key_point_templates[i] for i in key_point_indices]
            
            # Select a judge
            judge = np.random.choice(judge_templates)
            
            # Format date in DD/MM/YYYY format
            date = f"{(i % 28) + 1:02d}/{(i % 12) + 1:02d}/{case_year}"
            
            # Add the sample case with a decreasing similarity score
            similarity = 0.95 - (i * 0.08)
            if similarity < 0.5:
                similarity = 0.5 + (np.random.random() * 0.1)  # Add some randomness for lower scores
                
            samples.append({
                'title': title,
                'case_number': case_id,
                'date': date,
                'judges': judge,
                'summary': summary,
                'key_points': key_points,
                'similarity_score': similarity,
                'petitioner': petitioner,
                'respondent': respondent,
                'case_type': case_type['name']
            })
        return samples
    
    def _get_sample_search_results(self, query: str, count: int) -> List[Dict[str, Any]]:
        """
        Get sample search results for development/testing when no real index is available.
        
        Args:
            query: Search query string
            count: Number of results to return
            
        Returns:
            List of sample search results
        """
        import hashlib
        
        # Create deterministic samples based on query
        query_hash = hashlib.md5(query.encode()).hexdigest()
        seed_value = int(query_hash, 16) % (2**32)
        np.random.seed(seed_value)
        
        # Make query terms appear in results
        query_terms = query.lower().split()
        
        # Delhi High Court case types and standard formats 
        case_types = [
            {"name": "Writ Petition (Civil)", "code": "W.P.(C)"},
            {"name": "Criminal Appeal", "code": "Crl.A."},
            {"name": "Civil Appeal", "code": "C.A."},
            {"name": "Arbitration Petition", "code": "ARB.P."},
            {"name": "Company Petition", "code": "CO.PET."},
            {"name": "Original Side", "code": "O.S."},
            {"name": "Letters Patent Appeal", "code": "LPA"},
            {"name": "Regular Second Appeal", "code": "RSA"}
        ]
        
        # Legal topics commonly handled in Delhi High Court
        case_topics = [
            "Land Acquisition", "Service Matter", "Recovery of Money", 
            "Tenancy Dispute", "Constitutional Challenge", "Contract Enforcement", 
            "Intellectual Property Infringement", "Corporate Insolvency", 
            "Matrimonial Dispute", "Environmental Compliance"
        ]
        
        # Well-known Delhi High Court judges
        judge_templates = [
            "Hon'ble Justice Rajiv Shakdher",
            "Hon'ble Justice Pratibha M. Singh",
            "Hon'ble Justice Sanjeev Narula",
            "Hon'ble Justice Vibhu Bakhru",
            "Hon'ble Justice C. Hari Shankar",
            "Hon'ble Justice Navin Chawla",
            "Hon'ble Justice Rekha Palli",
            "Hon'ble Justice Subramonium Prasad",
            "Hon'ble Justice Yashwant Varma"
        ]
        
        # Generate realistic-looking sample search results
        samples = []
        for i in range(count):
            case_type = np.random.choice(case_types)
            case_topic = np.random.choice(case_topics)
            case_year = 2018 + (i % 6)  # Years between 2018-2023
            
            # Generate case number in proper Delhi HC format
            case_number = f"{10000+i}/{case_year}"
            case_id = f"{case_type['code']} {case_number}"
            
            # Generate realistic party names
            petitioners = [
                "Aditya Sharma", "Raj Enterprises Pvt. Ltd.", "Sunil Kumar", 
                "Delhi Residents Welfare Association", "Vikas Construction Co.", 
                "Kavita Gupta", "Modern School", "Nandini Singh", "Bharat Industries", 
                "Ravi Textiles Ltd.", "Indian Medical Association", "RK Builders"
            ]
            
            respondents = [
                "Union of India", "State of Delhi", "Delhi Development Authority",
                "Municipal Corporation of Delhi", "Delhi Electricity Regulatory Commission",
                "Delhi Metro Rail Corporation", "Delhi Urban Shelter Improvement Board",
                "All India Institute of Medical Sciences", "Delhi Transport Corporation",
                "Delhi Jal Board", "Delhi Police", "Income Tax Department"
            ]
            
            petitioner = petitioners[i % len(petitioners)]
            respondent = respondents[i % len(respondents)]
            
            # Create title that might include query terms
            title = f"{petitioner} vs {respondent}"
            if query_terms and i < 3:  # Ensure first few results have query terms in title
                term = np.random.choice(query_terms)
                title = f"{petitioner} (Re: {term.title()}) vs {respondent}"
            
            # Create a summary that includes query terms
            summary_templates = [
                "This case concerns {query_mention} where the petitioner sought relief against the decision of the authorities. The Delhi High Court examined whether the action was justified under the applicable legal framework.",
                "A {query_mention} matter brought before the Delhi High Court by {petitioner}, challenging orders passed by {respondent}. The Court analyzed the statutory provisions and precedents.",
                "This {case_type} filed by {petitioner} involves issues related to {query_mention}, with claims against {respondent} for alleged violations of statutory provisions.",
                "The Delhi High Court's judgment in this {query_mention} case established important principles regarding administrative actions and jurisdictional considerations.",
                "A significant ruling concerning {query_mention}, where the Court addressed questions of law raised by {petitioner} against actions of {respondent}."
            ]
            
            # Insert query terms into the summary
            if query_terms:
                term = np.random.choice(query_terms)
                query_mention = f"{term.lower()} and related matters"
            else:
                query_mention = f"{case_topic.lower()}"
                
            summary = np.random.choice(summary_templates).format(
                query_mention=query_mention,
                petitioner=petitioner,
                respondent=respondent,
                case_type=case_type['name']
            )
            
            # Select a judge
            judge = np.random.choice(judge_templates)
            
            # Format date in DD/MM/YYYY format
            date = f"{(i % 28) + 1:02d}/{(i % 12) + 1:02d}/{case_year}"
            
            # Add the sample result with a decreasing relevance score
            relevance = 0.95 - (i * 0.07)
            if relevance < 0.5:
                relevance = 0.5 + (np.random.random() * 0.1)  # Add some randomness for lower scores
                
            samples.append({
                'title': title,
                'case_number': case_id,
                'date': date,
                'judges': judge,
                'summary': summary,
                'relevance_score': relevance,
                'petitioner': petitioner,
                'respondent': respondent,
                'case_type': case_type['name']
            })
        return samples
