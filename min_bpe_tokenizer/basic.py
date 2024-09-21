from .base import MyTokenizer

from .utils import byte_pair_encoding, merge


class MyBasicTokenizer(MyTokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        # Implement training method for basic tokenizer
        num_of_merges = vocab_size - 256
        merges, vocab, _ = byte_pair_encoding(text, num_of_merges)
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        if len(tokens) >= 2:
            for pair, idx in self.merges.items():
                if len(tokens) < 2:
                    break
                tokens = merge(tokens, pair, idx)

        return tokens

    def decode(self, token_ids):
        token_bytes = b"".join(self.vocab[idx] for idx in token_ids)
        decoded_text = token_bytes.decode('utf-8', errors='replace')

        return decoded_text
