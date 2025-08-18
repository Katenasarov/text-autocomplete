import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.next_token_dataset import NextTokenDataset
from src.lstm_model import LSTMTextGenerator

def collate_fn(batch, pad_token_id=50256):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    return input_ids, labels

def train_lstm_model(train_loader, val_loader, vocab_size, device='cuda'):
    model = LSTMTextGenerator(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=50256)  # ignore pad/eos
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        scheduler.step()

    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("Модель сохранена: models/lstm_model.pth")
    return model