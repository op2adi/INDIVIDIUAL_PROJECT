# Import required libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Create the Sentiment Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create the Sentiment Classifier model class
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

# Training function
def train(model, train_dataloader, val_dataloader, num_epochs=3, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}')

        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    print('Training completed.')

# Evaluation function
def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted_labels = torch.argmax(outputs, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = total_correct / total_samples

    return avg_val_loss, val_accuracy

# Prediction function
def predict(model, text, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

    return probabilities

if __name__ == "__main__":
    # Generate toy data (replace with actual data loading and preprocessing)
    train_texts = [
        "I really enjoyed this movie. It was fantastic!",
        "The movie was terrible. I didn't like it at all.",
        "The food at this restaurant is amazing!",
        "The service was very slow, and the food wasn't good.",
        "This book is a masterpiece!",
    ]
    train_labels = [1, 0, 1, 0, 1]

    val_texts = [
        "The weather today is great!",
        "The product is of poor quality. Don't buy it.",
        "I had a wonderful experience at the event.",
    ]
    val_labels = [1, 0, 1]

    test_texts = [
        "The hotel room was clean and comfortable.",
        "I regret buying this product. It's not worth the money.",
    ]
    test_labels = [1, 0]

    # Initialize the tokenizer and create dataloaders for train, val, and test sets
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    max_length = 128
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the SentimentClassifier model
    num_classes = 2
    model = SentimentClassifier('distilbert-base-uncased', num_classes)

    # Train the model on CPU
    train(model, train_dataloader, val_dataloader, num_epochs=30, lr=2e-5)

    # Evaluate the model on CPU
    test_loss, test_accuracy = evaluate(model, test_dataloader, nn.CrossEntropyLoss(), 'cpu')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Example usage of predict function on CPU
    raw_text = "I really enjoyed this movie. It was fantastic!"
    probabilities = predict(model, raw_text, tokenizer)
    print(f'Class probabilities: {probabilities}')

