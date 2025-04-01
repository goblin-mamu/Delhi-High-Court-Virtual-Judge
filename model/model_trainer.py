import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LegalBERTTrainer:
    def __init__(
        self, 
        model_name="nlpaueb/legal-bert-base-uncased", 
        num_labels=5,  # Adjust based on your judgment categories
        output_dir="./fine_tuned_model"
    ):
        """
        Initialize the LegalBERT trainer.
        
        Args:
            model_name: Pre-trained model to use
            num_labels: Number of judgment categories
            output_dir: Directory to save the fine-tuned model
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        self.model.to(self.device)
    
    def prepare_data(self, judgments_df):
        """
        Prepare data for fine-tuning.
        
        Args:
            judgments_df: DataFrame with columns 'text' (judgment text) and 'label' (judgment category)
        
        Returns:
            train_dataset, eval_dataset: Datasets for training and evaluation
        """
        # Convert label strings to integers if needed
        if judgments_df['label'].dtype == 'object':
            label_map = {label: i for i, label in enumerate(judgments_df['label'].unique())}
            judgments_df['label_id'] = judgments_df['label'].map(label_map)
        else:
            judgments_df['label_id'] = judgments_df['label']
        
        # Split data into train and validation sets
        train_df, eval_df = train_test_split(judgments_df, test_size=0.2, random_state=42)
        
        # Create Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=512
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        
        # Set format for pytorch
        train_dataset.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "label_id"]
        )
        eval_dataset.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "label_id"]
        )
        
        return train_dataset, eval_dataset
    
    def train(self, train_dataset, eval_dataset, batch_size=8, num_epochs=3):
        """
        Fine-tune the LegalBERT model.
        
        Args:
            train_dataset: Dataset for training
            eval_dataset: Dataset for evaluation
            batch_size: Batch size for training
            num_epochs: Number of training epochs
        
        Returns:
            The fine-tuned model
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the model and tokenizer
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
        return self.model
    
    def _compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Tuple of predictions and labels
        
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == labels)
        
        return {
            "accuracy": accuracy
        }

def download_delhi_court_judgments():
    """
    Placeholder function to download and prepare Delhi High Court judgments.
    In a real implementation, this would connect to appropriate APIs or datasets.
    
    Returns:
        DataFrame with columns 'text' (judgment text) and 'label' (judgment category)
    """
    # In practice, this would involve:
    # 1. Downloading judgments from official sources or APIs
    # 2. Cleaning and preprocessing the text
    # 3. Categorizing judgments (e.g., allowed, dismissed, etc.)
    
    # This is a placeholder - in a real implementation, you would replace this
    print("Note: This is a placeholder. In production, download actual Delhi HC judgments")
    
    # Example structure (not real data)
    sample_data = {
        'text': [
            "This is a placeholder for judgment text 1...",
            "This is a placeholder for judgment text 2...",
            # More judgment texts would be here
        ],
        'label': [
            0,  # e.g., "Allowed"
            1,  # e.g., "Dismissed"
            # More labels would be here
        ]
    }
    
    return pd.DataFrame(sample_data)

def fine_tune_legal_bert():
    """
    Main function to fine-tune LegalBERT on Delhi High Court judgments.
    
    Returns:
        Path to the fine-tuned model
    """
    # Create output directory if it doesn't exist
    output_dir = "./fine_tuned_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and prepare Delhi High Court judgments
    judgments_df = download_delhi_court_judgments()
    
    # Initialize trainer
    trainer = LegalBERTTrainer(
        model_name="nlpaueb/legal-bert-base-uncased",
        num_labels=len(judgments_df['label'].unique()),
        output_dir=output_dir
    )
    
    # Prepare data
    train_dataset, eval_dataset = trainer.prepare_data(judgments_df)
    
    # Train model
    trainer.train(train_dataset, eval_dataset)
    
    return output_dir

if __name__ == "__main__":
    fine_tune_legal_bert()
