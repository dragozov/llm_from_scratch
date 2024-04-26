import re

TOKENIZER_RE = re.compile(r'([^\w<>|])')
SUB_RE = re.compile(r'\s+([^\w<>|])')
class Tokenizer(object):
    def __init__(self, training_data):
        tokens = self.tokenize(training_data)
        self.vocab, self.reversed = self.make_vocab(tokens)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab[token] if token in self.vocab else self.vocab['<|unk|>'] for token in tokens]

    def decode(self, encoded):
        text = ' '.join([self.reversed[i] for i in encoded])
        text = SUB_RE.sub(r'\1', text)
        return text

    def tokenize(self, text):
        result = []
        for token in TOKENIZER_RE.split(text):
            token = token.strip()
            if token:
                result.append(token)
        return result
    
    def make_vocab(self, tokens):
        tokens = sorted(list(set(tokens)))
        tokens.extend(['<|endoftext|>', '<|unk|>'])
        vocab = {}
        reversed = {}
        for i, token in enumerate(tokens):
            vocab[token] = i
            reversed[i] = token
        return vocab, reversed
