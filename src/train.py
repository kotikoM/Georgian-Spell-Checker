import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import time

from src.model import CharSeq2Seq, vocab_size, encode_word, PAD_IDX
from src.word_augmentation import get_corrupted_words

class SpellDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs  # list of (src_seq, tgt_seq)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)

    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_batch = torch.full((len(batch), max_src), PAD_IDX, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt), PAD_IDX, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_batch[i, :len(s)] = torch.tensor(s)
        tgt_batch[i, :len(t)] = torch.tensor(t)

    return src_batch, tgt_batch


words = open('../words/ganmarteba.ge_words.txt', 'r', encoding='utf-8').read().split('\n')
random.shuffle(words)

# pre-encode
dataset = [
    (encode_word(c), encode_word(w))
    for w in words
    for c in [w] + get_corrupted_words(w)
]
print(f'Generated {len(dataset)} corrupted pairs from {len(words)} words')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
print(f"Device: {device}")

model = CharSeq2Seq(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.001)

split_ratio = 0.8
split_idx = int(len(dataset) * split_ratio)

train_dataset = SpellDataset(dataset[:split_idx])
val_dataset = SpellDataset(dataset[split_idx:])

print(f"Training pairs: {len(train_dataset)}, Validation pairs: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=pin_memory
)

num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = 0

    model.train()
    for src, tgt in train_loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(
            output.view(-1, vocab_size),
            tgt.view(-1)
        )
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)


    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            teacher_forcing_ratio = max(0.5, 1.0 - epoch / num_epochs)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(
                output.view(-1, vocab_size),
                tgt.view(-1)
            )
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    epoch_time = time.time() - start_time
    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Time: {epoch_time:.2f}s"
    )

    # --- Save checkpoint -- optional ---
    checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# final save
final_model_path = "georgian_spellcheck_model_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved: {final_model_path}")
