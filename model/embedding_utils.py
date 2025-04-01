try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import numpy as np
import hashlib

class LegalDocumentEmbedder:
    def __init__(self, model_path="nlpaueb/legal-bert-base-uncased"):
        """
        Initialize the legal document embedder.
        
        Args:
            model_path: Path to the model (pre-trained or fine-tuned)
        """
        self.model_path = model_path
        self.dimension = 768  # Standard BERT embedding dimension
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()
                self.using_mock = False
                print(f"Using actual LegalBERT model from {model_path}")
            except Exception as e:
                print(f"Error loading transformers model: {str(e)}. Using mock embeddings.")
                self.using_mock = True
        else:
            self.using_mock = True
            print("Transformers library not available. Using mock embeddings.")
    
    def get_document_embedding(self, text, max_length=512, batch_size=8):
        """
        Generate an embedding for a legal document.
        For long documents, this splits the text into chunks, embeds each,
        and returns the average embedding.
        
        Args:
            text: The document text
            max_length: Maximum token length for each chunk
            batch_size: Batch size for processing chunks
            
        Returns:
            document_embedding: Numpy array of the document embedding
        """
        if self.using_mock:
            return self._get_mock_embedding(text)
            
        # Using actual transformer model
        # Split long documents into chunks of approximately max_length tokens
        tokens = self.tokenizer.tokenize(text)
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        
        # Process chunks in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [self.tokenizer.convert_tokens_to_string(chunk) for chunk in batch_chunks]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token embedding as the representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batch embeddings and average them
        all_embeddings = np.vstack(all_embeddings)
        document_embedding = np.mean(all_embeddings, axis=0)
        
        # Normalize the embedding to unit length
        document_embedding = document_embedding / np.linalg.norm(document_embedding)
        
        return document_embedding
    
    def get_embeddings_batch(self, texts, max_length=512):
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum token length
            
        Returns:
            embeddings: Numpy array of embeddings, one per text
        """
        if self.using_mock:
            embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, text in enumerate(texts):
                embeddings[i] = self._get_mock_embedding(text)
            return embeddings
            
        # Using actual transformer model
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding as the representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize the embeddings to unit length
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
        
    def _get_mock_embedding(self, text):
        """
        Generate a deterministic mock embedding based on the text content.
        This is used when transformers library is not available.
        
        Args:
            text: The document text
            
        Returns:
            mock_embedding: A deterministic mock embedding vector
        """
        # Create a deterministic seed based on text content
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed_value = int(text_hash, 16) % (2**32)
        np.random.seed(seed_value)
        
        # Generate a random vector
        mock_embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
        
        # Normalize to unit length
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        
        return mock_embedding
