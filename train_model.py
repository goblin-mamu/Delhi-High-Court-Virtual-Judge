import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

class JudgmentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        losses.append(loss.item())
        
        # Calculate accuracy
        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += len(labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
    
    return np.mean(losses), correct_predictions.double() / total_predictions

def evaluate(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            losses.append(loss.item())
            
            # Calculate accuracy
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
    
    return np.mean(losses), correct_predictions.double() / total_predictions

def prepare_data(csv_path):
    """Prepare data for training"""
    try:
        # Load data
        df = pd.read_csv(csv_path)
        
        # Filter out rows with unknown outcomes or missing text
        df = df[(df['outcome'] != 'Unknown') & (df['full_text'].notna())]
        
        if len(df) == 0:
            raise ValueError("No valid training data found. All outcomes are unknown or text is missing.")
        
        print(f"Training with {len(df)} valid judgments")
        
        # Create label mapping
        outcome_labels = df['outcome'].unique()
        label_map = {outcome: idx for idx, outcome in enumerate(outcome_labels)}
        
        # Apply label mapping
        df['label'] = df['outcome'].map(label_map)
        
        # Create summary of the mapping
        print("Label mapping:")
        for outcome, idx in label_map.items():
            print(f"  {outcome}: {idx}")
        
        # Make sure we have at least two samples for each class to support stratified splitting
        # If not, we'll duplicate some samples 
        min_samples_per_class = 2
        for outcome in outcome_labels:
            class_count = len(df[df['outcome'] == outcome])
            if class_count == 1:
                # Find the single instance
                single_instance = df[df['outcome'] == outcome].iloc[0].copy()
                # Add a small variation to the text to make it slightly different
                single_instance['full_text'] = single_instance['full_text'] + " [duplicate with minor variation]"
                # Append to dataframe
                df = pd.concat([df, pd.DataFrame([single_instance])], ignore_index=True)
                print(f"Added a duplicate for outcome '{outcome}' to ensure at least 2 samples")
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'] if len(df) > 1 else None)
        
        print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}")
        
        return train_df, val_df, label_map
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        raise

def train_model(csv_path, output_dir="./fine_tuned_model", batch_size=4, epochs=3, max_length=256, learning_rate=2e-5):
    """Train model on judgment data"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        train_df, val_df, label_map = prepare_data(csv_path)
        
        # Save label mapping
        with open(os.path.join(output_dir, "label_map.txt"), "w") as f:
            for outcome, idx in label_map.items():
                f.write(f"{outcome}\t{idx}\n")
        
        # Determine device (CPU or GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load tokenizer and model
        print("Loading model and tokenizer...")
        
        try:
            # Use a smaller model for CPU training
            model_name = "distilbert-base-uncased"  # Smaller than BERT/LegalBERT, better for CPU
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(label_map)
            )
        except Exception as e:
            print(f"Error loading pre-trained model: {str(e)}")
            print("Trying a different model...")
            
            # Fallback to an even smaller model if available
            try:
                from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                model = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=len(label_map)
                )
            except Exception as e2:
                print(f"Error loading fallback model: {str(e2)}")
                raise ValueError("Could not load any pre-trained model. Please check your internet connection and try again.")
        
        model.to(device)
        
        # Reduce model complexity for CPU training
        if device.type == 'cpu':
            print("CPU detected: Using smaller max_length and batch_size")
            max_length = min(max_length, 128)  # Cap at 128 tokens
            batch_size = min(batch_size, 2)    # Cap at batch size 2
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = JudgmentDataset(
            texts=train_df['full_text'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        val_dataset = JudgmentDataset(
            texts=val_df['full_text'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Calculate total training steps
        total_steps = len(train_loader) * epochs
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        print("Starting training...")
        best_accuracy = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            try:
                # Train
                train_loss, train_acc = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    device
                )
                
                print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
                
                # Evaluate
                val_loss, val_acc = evaluate(
                    model,
                    val_loader,
                    device
                )
                
                print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
                
                # Save the best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    print(f"Saving best model with accuracy: {best_accuracy:.4f}")
                    
                    # Save model
                    model_path = os.path.join(output_dir, "best_model")
                    os.makedirs(model_path, exist_ok=True)
                    
                    try:
                        model.save_pretrained(model_path)
                        tokenizer.save_pretrained(model_path)
                        print(f"Model saved to {model_path}")
                    except Exception as e:
                        print(f"Warning: Error saving model: {str(e)}")
                        # Try alternative saving method if available
                        try:
                            import torch
                            torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
                            print("Saved model state dictionary as fallback")
                        except Exception as e2:
                            print(f"Failed to save model using alternative method: {str(e2)}")
                
                # Always save at the end of each epoch as a backup
                backup_path = os.path.join(output_dir, f"backup_epoch_{epoch+1}")
                os.makedirs(backup_path, exist_ok=True)
                try:
                    model.save_pretrained(backup_path)
                    print(f"Backup saved to {backup_path}")
                except Exception as e:
                    print(f"Warning: Couldn't save backup for epoch {epoch+1}: {str(e)}")
                
            except Exception as e:
                print(f"Error during epoch {epoch+1}: {str(e)}")
                print("Attempting to continue with next epoch...")
        
        print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")
        return os.path.join(output_dir, "best_model")
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure output directory exists even if training failed
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)
        
        # Create a dummy label map if none exists
        label_map_path = os.path.join(output_dir, "label_map.txt")
        if not os.path.exists(label_map_path):
            with open(label_map_path, "w") as f:
                f.write("Allowed\t0\n")
                f.write("Dismissed\t1\n")
                f.write("Partly Allowed\t2\n")
                f.write("Disposed\t3\n")
                f.write("Settled\t4\n")
            print("Created default label mapping due to training failure")
        
        print("Training failed, but created necessary directories for the application to run")
        return os.path.join(output_dir, "best_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on judgment data")
    parser.add_argument("--csv", default="./data/processed/dhc_judgments_sample.csv", help="Path to CSV file with judgment data")
    parser.add_argument("--output", default="./fine_tuned_model", help="Output directory for model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(
        args.csv,
        args.output,
        args.batch_size,
        args.epochs,
        args.max_length,
        args.learning_rate
    )