class MyTokenizer:
    def __init__(self):
        self.merges = {}
        self.special_tokens = {}
        self.pattern = ""
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, token_ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (pair0, pair1), idx in self.merges.items():
            vocab[idx] = vocab[pair0] + vocab[pair1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def save_model(self, file_prefix):
        # Save merges to file
        model_filename = file_prefix + '.model'
        with open(model_filename, 'w') as out_file:
            out_file.write('minbpe v1\n')
            out_file.write(f'{self.pattern}\n')
            out_file.write(f"{len(self.special_tokens)}\n")
            for special_token, idx in self.special_tokens.items():
                out_file.write(f'{special_token} {idx}\n')
            for idx1, idx2 in self.merges:
                out_file.write(f'{idx1} {idx2}\n')

    def load_model(self, filename):
        merges = {}
        idx = 256
        with open(filename, 'r') as in_file:
            bpe_version = in_file.readline()  # skip bpe version
            self.pattern = in_file.readline().strip()
            num_spec_tokens = int(in_file.readline().strip())
            # Read special tokens from file
            for _ in range(num_spec_tokens):
                spec_token, special_tok_idx = in_file.readline().strip().split()
                self.special_tokens[spec_token] = int(special_tok_idx)
            # Read merges from file
            for line in in_file.readlines():
                idx1, idx2 = map(int, line.rstrip().split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab
