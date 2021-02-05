import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from _collections import defaultdict

class Vocabulary(object):
    def __init__(self):
        self.label2id = {"neutral": 0,
                         "surprise": 1,
                         "fear": 2,
                         "sadness": 3,
                         "joy": 4,
                         "anger": 5,
                         "disgust": 6}
        self.id2label = {value: key for key, value in self.label2id.items()}

    def num_labels(self):
        return len(self.label2id)


def collate_fn(data):
    dia_input, emo_label, spk_label, cls_index = map(list, zip(*data))

    batch_size = len(dia_input)
    max_dia_len = np.max([x.shape[0] for x in emo_label])

    emo_mask = torch.zeros((batch_size, max_dia_len), dtype=torch.bool)
    spk_mask = torch.zeros((batch_size, max_dia_len, max_dia_len), dtype=torch.bool)
    _spk_label = torch.zeros((batch_size, max_dia_len, max_dia_len), dtype=torch.long)
    for i, (e, s) in enumerate(zip(emo_label, spk_label)):
        emo_mask[i, :e.shape[0]] = True
        spk_mask[i, :s.shape[0], :s.shape[1]] = True
        _spk_label[i, :s.shape[0], :s.shape[1]] = s
    spk_label = _spk_label

    dia_input = pad_sequence(dia_input, True)
    emo_label = pad_sequence(emo_label, True)
    cls_index = pad_sequence(cls_index, True)

    return dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask


class MyDataset(Dataset):
    def __init__(self, dia_input, emo_label, spk_label, cls_index):
        self.dia_input = dia_input
        self.emo_label = emo_label
        self.spk_label = spk_label
        self.cls_index = cls_index

    def __getitem__(self, item):
        return torch.LongTensor(self.dia_input[item]), \
               torch.LongTensor(self.emo_label[item]), \
               torch.LongTensor(self.spk_label[item]), \
               torch.LongTensor(self.cls_index[item])

    def __len__(self):
        return len(self.dia_input)


def load_data(input_max):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = Vocabulary()

    def processing_data(path):
        data = pd.read_csv(path, encoding="utf-8")
        dia_dict = defaultdict(list)
        for i, item in data.iterrows():
            utt_data = {"utterance": item["Utterance"],
                        "emotion": item["Emotion"],
                        "speaker": item["Speaker"]}
            dia_dict[item["Dialogue_ID"]].append(utt_data)

        dia_input = []
        cls_index = []
        emo_label = []
        spk_label = []
        for dia in dia_dict.values():

            _dia_input = [tokenizer.encode(x['utterance'])[:-1] for x in dia]
            _len_index = np.cumsum([len(x) for x in _dia_input])
            chucks = 1
            for i in range(len(_len_index) - 1):
                start = _len_index[i]
                end = _len_index[i + 1]
                if start < input_max * chucks < end:
                    _dia_input[i] += [tokenizer.pad_token_id] * (input_max - start)
                    _len_index = np.cumsum([len(x) for x in _dia_input])
                    chucks += 1

            _dia_input = [x for utt in _dia_input for x in utt] + [tokenizer.sep_token_id]
            _cls_index = [i for i, x in enumerate(_dia_input) if x == tokenizer.cls_token_id]
            _emo_label = [vocab.label2id[x["emotion"]] for x in dia]
            assert len(_emo_label) == len(_cls_index)

            _spk_label = np.zeros((len(dia), len(dia)), dtype=np.long)
            for i in range(len(dia)):
                for j in range(len(dia)):
                    _spk_label[i, j] = dia[i]["speaker"] == dia[j]["speaker"]

            dia_input.append(_dia_input)
            emo_label.append(_emo_label)
            spk_label.append(_spk_label)
            cls_index.append(_cls_index)

        return MyDataset(dia_input, emo_label, spk_label, cls_index)

    return (
               processing_data('./data/train_sent_emo.csv'),
               processing_data('./data/dev_sent_emo.csv'),
               processing_data('./data/test_sent_emo.csv')
           ), vocab