import torch
import torch.nn as nn
from transformers import BertModel
from models.Trans import TransformerEncoder


class HiTrans(nn.Module):
    def __init__(self, hidden_dim, emotion_class_num, d_model, d_ff, heads, layers, dropout=0, input_max_length=512):
        super(HiTrans, self).__init__()
        self.input_max_length = input_max_length
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.encoder = TransformerEncoder(d_model, d_ff, heads, layers, 0.1)

        self.emo_output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emotion_class_num)
        )

        self.spk_output_layer = SpeakerMatchLayer(hidden_dim, dropout)

    def forward(self, dia_input, cls_index, mask):
        bert_outputs = []
        for i in range((dia_input.size(1) - 1) // self.input_max_length + 1):
            cur_input = dia_input[:, i * self.input_max_length:(i + 1) * self.input_max_length]
            cur_mask = cur_input.ne(0)
            bert_output, _ = self.bert(cur_input, cur_mask)
            bert_outputs.append(bert_output)
        bert_outputs = torch.cat(bert_outputs, dim=1)

        bert_outputs = bert_outputs[torch.arange(bert_outputs.size(0)).unsqueeze(1), cls_index.long()]
        bert_outputs = bert_outputs * mask[:, :, None].float()

        bert_outputs = self.encoder(bert_outputs, mask)
        emo_output = self.emo_output_layer(bert_outputs)
        spk_output = self.spk_output_layer(bert_outputs)

        return emo_output, spk_output


class SpeakerMatchLayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(SpeakerMatchLayer, self).__init__()
        self.mpl1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.mpl2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.biaffine = Biaffine(hidden_dim // 2, 2)

    def forward(self, x):
        x1 = self.mpl1(x)
        x2 = self.mpl2(x)
        output = self.biaffine(x1, x2)
        return output


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)
        return s
