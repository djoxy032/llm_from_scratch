def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids[:-1], ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def byte_pair_encoding(text, num_merges):
    tokens = text.encode('utf-8', errors='replace')
    tokens_processed = list(map(int, tokens))
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = {}
    for i in range(num_merges):
        stats = get_stats(tokens_processed)
        max_pair = max(stats, key=stats.get)
        tokens_processed = merge(tokens_processed, max_pair, 256 + i)
        vocab[256 + i] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges[max_pair] = 256 + i

    return merges, vocab, tokens_processed
