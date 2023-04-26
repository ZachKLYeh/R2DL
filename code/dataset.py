import torch
from torch.utils.data import Dataset
from Bio import SeqIO

def read_fasta(path):
    sequences = list(SeqIO.parse(path, format="fasta"))
    res = []
    for sequence in sequences:
        res.append(str(sequence.seq))
    return res


def tokenizer(seq):
    table = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    tokens = [0]*128
    attention_mask = [0]*128
    for i,x in enumerate(seq[0:len(seq)]):
        if i < 128:
            tokens[i] = table.index(x) + 1
            attention_mask[i] = 1
    return tokens, attention_mask


class AAPDataset(Dataset):
    def __init__(self, mode = "train", data = "NT15"):
        self.tokenized_seqs = []
        self.attention_masks = []
        self.labels = []

        if mode == "train":
            if data == "NT15":
                pos_path = "../dataset/train_NT15_pos_80.fasta"
                neg_path = "../dataset/train_NT15_neg_80.fasta"

            else:
                pos_path = "../../ACPNet/data/AAPs106.txt"
                neg_path = "../../ACPNet/data/non-AAPs106.txt"
        else:
            if data == "NT15":
                pos_path = "../dataset/test_NT15_pos_19.fasta"
                neg_path = "../dataset/test_NT15_neg_21.fasta"

            else:
                pos_path = "../../ACPNet/data/AAPs26.txt"
                neg_path = "../../ACPNet/data/non-AAPs26.txt"

        pos_seqs = read_fasta(pos_path)
        neg_seqs = read_fasta(neg_path)

        for seq in pos_seqs:
            tokens, attention_mask = tokenizer(seq)
            self.tokenized_seqs.append(tokens)
            self.attention_masks.append(attention_mask)
            self.labels.append([1, 0])
        for seq in neg_seqs:
            tokens, attention_mask = tokenizer(seq)
            self.tokenized_seqs.append(tokens)
            self.attention_masks.append(attention_mask)
            self.labels.append([0, 1])

    def __getitem__(self, index):
        item = {}
        item['input_ids'] = torch.tensor(self.tokenized_seqs[index])
        item['attention_mask'] = torch.tensor(self.attention_masks[index])
        item['labels'] = torch.Tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)

class AMPDataset(Dataset):
    def __init__(self):
        self.tokenized_seqs = []
        self.attention_masks = []
        self.labels = []
        pos_path = "../dataset/train_AMP_3268.fasta"
        neg_path = "../dataset/train_nonAMP_9777.fasta"
        pos_seqs = read_fasta(pos_path)
        neg_seqs = read_fasta(neg_path)
        for seq in pos_seqs:
            tokens, attention_mask = tokenizer(seq)
            self.tokenized_seqs.append(tokens)
            self.attention_masks.append(attention_mask)
            self.labels.append([1, 0])
        for i in range(3268):
            seq = neg_seqs[i]
            tokens, attention_mask = tokenizer(seq)
            self.tokenized_seqs.append(tokens)
            self.attention_masks.append(attention_mask)
            self.labels.append([0, 1])

    def __getitem__(self, index):
        item = {}
        item['input_ids'] = torch.tensor(self.tokenized_seqs[index])
        item['attention_mask'] = torch.tensor(self.attention_masks[index])
        item['labels'] = torch.Tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)
