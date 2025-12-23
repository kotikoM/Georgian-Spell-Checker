import random

import torch
from torch import nn

GEORGIAN_LETTERS = list("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ-")
special_tokens = ["<PAD>", "<SOS>", "<EOS>"]

vocab = special_tokens + GEORGIAN_LETTERS
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}
PAD_IDX = char2idx["<PAD>"]
vocab_size = len(vocab)


def encode_word(word):
    seq = [char2idx["<SOS>"]] + [char2idx[c] for c in word] + [char2idx["<EOS>"]]
    return seq


def decode_sequence(seq):
    chars = [idx2char[i] for i in seq if idx2char[i] not in ["<SOS>", "<EOS>"]]
    return "".join(chars)


class CharSeq2Seq(nn.Module):

    def __init__(self, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)

        # decode
        batch_size = src.size(0)
        max_len = tgt.size(1) if tgt is not None else 30
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)

        input_token = torch.full((batch_size, 1), char2idx["<SOS>"], dtype=torch.long).to(src.device)

        for t in range(max_len):
            embedded_input = self.embedding(input_token)
            output, hidden = self.decoder(embedded_input, hidden)
            output = self.fc(output)

            pred = output.argmax(2)
            outputs[:, t, :] = output.squeeze(1)

            if tgt is not None and random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t].unsqueeze(1)
            else:
                input_token = pred

            if tgt is None:
                if (pred == char2idx["<EOS>"]).all():
                    break

        return outputs

    @torch.no_grad()
    def correct_word(self, word: str, max_len: int = 50) -> str:
        """Greedy decoding for a single word, stops at <EOS>."""
        self.eval()
        src = torch.tensor([encode_word(word)], device=next(self.parameters()).device)
        embedded_src = self.embedding(src)

        # encode
        _, hidden = self.encoder(embedded_src)

        decoder_input = torch.tensor([[char2idx["<SOS>"]]], device=src.device)
        decoded_chars = []

        for _ in range(max_len):
            embedded_input = self.embedding(decoder_input)  # shape (1,1,embed_dim)
            output, hidden = self.decoder(embedded_input, hidden)
            output_logits = self.fc(output)
            next_token = output_logits.argmax(dim=-1)  # shape (1,1)
            token_id = next_token.item()

            if token_id == char2idx["<EOS>"]:
                break

            # skip <SOS> in output
            if token_id != char2idx["<SOS>"]:
                decoded_chars.append(idx2char[token_id])

            decoder_input = next_token.view(1, -1)  # enforce (batch_size, seq_len) for next step

        return "".join(decoded_chars)
