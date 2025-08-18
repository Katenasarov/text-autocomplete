import torch
import torch.nn as nn

class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (B, L, E)
        lstm_out, hidden = self.lstm(x, hidden)  # (B, L, H)
        logits = self.fc(self.dropout(lstm_out))  # (B, L, V)
        return logits, hidden

    def generate(self, tokenizer, prompt, max_length=20, device='cpu'):
        self.eval()
        with torch.no_grad():
            tokens = tokenizer.encode(prompt.lower(), return_tensors='pt').to(device)
            generated = tokens.clone()

            for _ in range(max_length - tokens.size(1)):
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :]  # последний токен
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            return tokenizer.decode(generated[0], skip_special_tokens=True)