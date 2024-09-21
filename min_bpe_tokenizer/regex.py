import regex as re
from .base import MyTokenizer
from .utils import get_stats, merge

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class MyRegexTokenizer(MyTokenizer):

    def __init__(self, pattern=None):
        super().__init__()

        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        num_merges = vocab_size - 256

        chunks = re.findall(self.compiled_pattern, text)

        ids = [list(word.encode('utf-8')) for word in chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            idx = 256 + i
            stats = {}

            # Get stats for all separate 'words'
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            # Get most frequent pair (max_pair)
            max_pair = max(stats, key=stats.get)

            # Update vocab
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]

            # Update ids in 'words to merge those ready for swap/merge
            ids = [merge(chunk_ids, max_pair, idx) for chunk_ids in ids]
            # Update merges with max_pair
            merges[max_pair] = idx

        self.vocab = vocab
        self.merges = merges

    def register_special_tokens(self, special_tokens):
        # word -> id
        self.special_tokens = special_tokens
        # id -> word used for decode
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):

        out_bytes = []
        for idx in ids:
            if idx in self.vocab:
                out_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                # inverse_special_tokens contain strings for values
                special_token_bytes = self.inverse_special_tokens[idx].encode(
                    'utf-8')
                out_bytes.append(special_token_bytes)
            else:
                raise ValueError('Invalid token')

        out_bytes = b"".join(out_bytes)
        out_text = out_bytes.decode('utf-8', errors='replace')

        return out_text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # skip if pair == inf
            if pair not in self.merges:
                break  # nothing else can be merged anymore

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items()
                       if k in allowed_special}
        else:
            raise ValueError(
                f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
