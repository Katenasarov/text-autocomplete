from rouge_score import rouge_scorer
from tqdm import tqdm
import torch

def evaluate_lstm(model, dataloader, tokenizer, device='cpu'):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    total_rouge1 = 0
    total_rouge2 = 0
    count = 0

    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Evaluating LSTM"):
            input_ids = input_ids.to(device)
            for i in range(input_ids.size(0)):
                prompt_ids = input_ids[i, :input_ids.size(1)//2]  # первая половина
                prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                true_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)

                generated = model.generate(tokenizer, prompt, max_length=30, device=device)
                scores = scorer.score(true_text, generated)
                total_rouge1 += scores['rouge1'].fmeasure
                total_rouge2 += scores['rouge2'].fmeasure
                count += 1

    print(f"LSTM ROUGE-1: {total_rouge1/count:.4f}")
    print(f"LSTM ROUGE-2: {total_rouge2/count:.4f}")
    return total_rouge1 / count, total_rouge2 / count