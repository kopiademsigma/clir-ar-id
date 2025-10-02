# dprx_train.py
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel, AdamW

# ---------------------------
# Config / Hyperparameters
# ---------------------------
MODEL_NAME = "xlm-roberta-base"   # multilingual backbone
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LEN = 160
LR = 2e-5
EPOCHS = 3
NUM_NEGATIVES = 3     # number of negatives sampled per query from its 'negatives' list
SEED = 42
SAVE_DIR = Path("./dprx_checkpoint")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
# ---------------------------


random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------
# Dataset
# ---------------------------
class JsonlTriplesDataset(Dataset):
    def __init__(self, path: str):
        self.path = Path(path)
        self.examples = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line.strip())
                # Expect: {"query": "...", "positives": ["..."], "negatives": ["...","..."]}
                query = obj.get("query") or obj.get("q")
                positives = obj.get("positives") or obj.get("positive") or []
                negatives = obj.get("negatives") or obj.get("negative") or []
                # keep as lists
                if not isinstance(positives, list):
                    positives = [positives]
                if not isinstance(negatives, list):
                    negatives = [negatives]
                # skip if no positives
                if not query or len(positives) == 0:
                    continue
                self.examples.append({"query": query, "positives": positives, "negatives": negatives})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the raw lists; sampling / tokenization done in collate
        return self.examples[idx]


# ---------------------------
# Collator: sample one positive + k negatives, tokenize batch
# ---------------------------
class TripletCollator:
    def __init__(self, tokenizer: XLMRobertaTokenizerFast, max_len=MAX_LEN, num_negatives=NUM_NEGATIVES):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_negatives = num_negatives

    def __call__(self, batch: List[Dict]):
        """
        batch: list of examples where each example is {'query': str, 'positives': [..], 'negatives': [..]}
        Output: tokenized tensors for queries, passages (positives+negatives stacked)
        """
        queries = []
        passages = []  # will be flattened: for batch size B and k negatives -> B*(1+k) passages
        positive_indices = []  # index of positive for each query in the passages list

        for ex_i, ex in enumerate(batch):
            q = ex["query"]
            # sample one positive randomly (you can choose 0 or max strategy instead)
            pos = random.choice(ex["positives"])

            # sample negatives (if not enough negatives, sample from other examples later)
            negs = ex["negatives"]
            if len(negs) >= self.num_negatives:
                sampled_negs = random.sample(negs, self.num_negatives)
            else:
                # if not enough negatives, repeat random choice (or you can sample in-batch negatives later)
                sampled_negs = random.choices(negs, k=self.num_negatives) if negs else [""] * self.num_negatives

            # append
            queries.append(q)
            positive_indices.append(len(passages))  # position where pos will be
            passages.append(pos)
            passages.extend(sampled_negs)

        # Tokenize queries and passages
        q_tok = self.tokenizer(queries, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        p_tok = self.tokenizer(passages, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        return q_tok, p_tok, positive_indices, len(batch)  # return batch size for convenience


# ---------------------------
# Encoder wrapper (shared encoder)
# ---------------------------
class DPRXEncoder(torch.nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # last_hidden_state[:,0,:] as representation
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (batch, dim)
        # normalize
        cls = F.normalize(cls, p=2, dim=-1)
        return cls


# ---------------------------
# Loss: InfoNCE / cross-entropy over (positive | negatives + in-batch negatives)
# Simpler approach: use per-batch concatenation of all passages, compute logits, labels are positions of positives
# ---------------------------
def dprx_step(encoder: DPRXEncoder, q_tok, p_tok, pos_indices, batch_size, device):
    # Move tokens to device
    q_input_ids = q_tok["input_ids"].to(device)
    q_attention_mask = q_tok["attention_mask"].to(device)
    p_input_ids = p_tok["input_ids"].to(device)
    p_attention_mask = p_tok["attention_mask"].to(device)

    # Encode
    q_vecs = encoder(q_input_ids, q_attention_mask)            # (B, D)
    p_vecs = encoder(p_input_ids, p_attention_mask)            # (B*(1+k), D)

    # Compute similarity matrix: (B, P) where P = B*(1+k)
    logits = torch.matmul(q_vecs, p_vecs.T)  # inner product; since normalized, same as cosine

    # Labels: for each query i, positive index is pos_indices[i]
    # But pos_indices were absolute positions in the passages list across the batch.
    # Need to convert to indices 0..P-1 mapped per query.
    # In our collator pos_indices already represent positions in passages list sequentially.
    labels = torch.tensor(pos_indices, dtype=torch.long, device=device)

    # Cross-entropy
    loss = F.cross_entropy(logits, labels)
    return loss, logits, q_vecs, p_vecs


# ---------------------------
# Training runner
# ---------------------------
def train_jsonl(
    jsonl_path: str,
    model_name=MODEL_NAME,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=DEVICE,
):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
    dataset = JsonlTriplesDataset(jsonl_path)
    collator = TripletCollator(tokenizer, max_len=MAX_LEN, num_negatives=NUM_NEGATIVES)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=False)

    encoder = DPRXEncoder(model_name).to(device)
    optimizer = AdamW(encoder.parameters(), lr=lr)

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0.0
        for step, (q_tok, p_tok, pos_indices, bsize) in enumerate(dataloader):
            loss, _, _, _ = dprx_step(encoder, q_tok, p_tok, pos_indices, bsize, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

        avg = total_loss / (step + 1)
        print(f"Epoch {epoch+1} finished — avg loss: {avg:.4f}")

        # Optionally save checkpoint each epoch
        save_path = SAVE_DIR / f"epoch-{epoch+1}"
        save_path.mkdir(parents=True, exist_ok=True)
        # save whole model & tokenizer
        encoder.backbone.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")

    return encoder, tokenizer


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # path to your jsonl file with lines like you pasted
    TRAIN_JSONL = "jh-polo/triplet.jsonl"
    encoder, tokenizer = train_jsonl(TRAIN_JSONL)
