import torch


def evaluate_model(model, tokenizer, dataset, max_length=128):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for item in dataset:
            text = item.get("sentence", item.get("text", ""))
            if not text and "prompt" in item:
                text = item["prompt"]

            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()
