# Georgian Character-Level Spellchecker

This project implements a character-level Sequence-to-Sequence (Seq2Seq) neural network designed to correct spelling
errors in the Georgian language.

Because Georgian features a unique alphabet with words being composed of root, prefixes and suffixes. It has 1:1
sound-to-letter mapping and no uppercase forms, it is an ideal candidate for a model that learns orthographic rules at
the character level.

---

## Project Overview

The model is built using a 2-layer LSTM architecture (Encoder-Decoder) that processes words in isolation.
It is trained to map "corrupted" Georgian words - simulating real-world typing errors like deletions, duplications,
swaps, and substituions.

## Quick Start: Inference

To use the trained model, you can load the final state dictionary and run the correct_word method.

```python
import torch
from src.model import CharSeq2Seq

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharSeq2Seq().to(device)
model.load_state_dict(torch.load("trained/georgian_spellcheck_model_final.pth"))

# Example usage
print(model.correct_word("ფანჯრა"))  # Output: ფანჯარა
```

## Conclusion

While the model is not perfect, it successfully demonstrates how a character-level Seq2Seq architecture can master
Georgian orthography and repair common human errors.
By learning the language's consistent phonetic patterns, it can recover complex roots even from significant corruption