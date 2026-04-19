import torch


class dna_one_hot:
    """we keep the encoder tiny because the project only needs one stable representation"""

    def __init__(self):
        # we send gaps and anything unexpected to the same channel so odd FASTA symbols
        # don't quietly create fake nucleotides downstream
        self.labels = {'a': 0, 'c': 1, 'g': 2, 't': 3, '-': 4}

    def _encode_seq(self, batch):
        one_hot_seq = []
        for seq in batch['sequence']:
            encoding = [self.labels.get(char, 4) for char in seq.lower()]
            one_hot = torch.zeros((len(encoding), len(self.labels)))
            one_hot[torch.arange(len(encoding)), encoding] = 1
            one_hot_seq.append(one_hot.T)
        return {'input_ids': torch.stack(one_hot_seq)}

    def __call__(self, batch):
        return self._encode_seq(batch)
