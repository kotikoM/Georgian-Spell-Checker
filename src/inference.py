import torch

from src.model import CharSeq2Seq, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharSeq2Seq(vocab_size)
model.load_state_dict(torch.load("georgian_spellcheck_model_final.pth", map_location=device))

# checkpoint = torch.load("checkpoint_epoch_1.pth", map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
print("Model loaded")

test_words = [
    "თბილისი",
    "მთვარეე",
    "ცამოდი",
    "ქიგნი",

]

print("Spellcheck results:\n")
for w in test_words:
    corrected = model.correct_word(w)
    print(f"{w} -> {corrected}")