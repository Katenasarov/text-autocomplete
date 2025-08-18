from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=20):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.examples = []

        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding=False)
            input_ids = encodings['input_ids']
            if len(input_ids) > 1:  # нужно минимум 2 токена
                self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        # X: всё кроме последнего токена
        input_ids_x = input_ids[:-1]
        # Y: следующие токены (сдвиг)
        labels = input_ids[1:]
        return {
            'input_ids': input_ids_x,
            'labels': labels
        }