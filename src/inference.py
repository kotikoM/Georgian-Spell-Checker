import torch

from src.model import CharSeq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharSeq2Seq()
model.load_state_dict(torch.load("../trained/georgian_spellcheck_model_final.pth", map_location=device))

# for checkpoint loading
# checkpoint = torch.load("checkpoint_epoch_1.pth", map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
print("Model loaded")

test_words = [
    "გამარჯობა", # correct
    "რობოტი",  # correct
    "გიორგი",  # correct
    "გზა", # correct
    "მწვრთნელი", # correct
    "იასამანი",  # correct
    "ჰიდროელექტროსადგური",  # correct

    "გამარჯობ", # deleted ი
    "მზეზ",  # deleted ი
    "სოკ",  # deleted ო
    "გამ",  # deleted ო
    "ფანჯრა",  # deleted ა

    "მთვარეე", # duplicated ე
    "მზეზზე",  # duplicated ზ
    "მგელიი",  # duplicated ი
    "წიგნნნი",  # triplicated ნ
    "გოგონნა",  # duplicated ნ

    "ცამოდი", # appended ც
    "დავთით", # swapped თ ვ
    "ყტემალი", # swapped ტ ყ
    "მოტოციკკეტი", # swapped ლ to კ

    "სოო",  # many words could form, model chose სოლო
    "" # just curious on output for empty string, model chose ალ
]

print("Spellcheck results:\n")
for w in test_words:
    corrected = model.correct_word(w)
    print(f"-    {w} -> {corrected}")
