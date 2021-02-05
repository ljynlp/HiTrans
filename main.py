import argparse
import time
import utils
import dataset
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from models.HiTrans import HiTrans
from models.Loss import MultiTaskLoss
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import random
import numpy as np


class Trainer(object):
    def __init__(self, model):
        self.model = model.cuda()

        self.emo_criterion = nn.CrossEntropyLoss()
        self.spk_criterion = nn.CrossEntropyLoss()

        self.multi_loss = MultiTaskLoss(2).cuda()
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': args.bert_lr,
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': args.bert_lr,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': args.lr,
             'weight_decay': args.weight_decay},
            {"params": self.multi_loss.parameters(),
             'lr': args.lr,
             "weight_decay": args.weight_decay}
        ]

        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.alpha)

    def train(self, data_loader):
        self.model.train()
        loss_array = []
        emo_gold_array = []
        emo_pred_array = []
        for dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask in data_loader:
            dia_input = dia_input.cuda()
            emo_label = emo_label.cuda()
            spk_label = spk_label.cuda()
            cls_index = cls_index.cuda()
            emo_mask = emo_mask.cuda()
            spk_mask = spk_mask.cuda()

            emo_output, spk_output = self.model(dia_input, cls_index, emo_mask)

            emo_output = emo_output[emo_mask]
            emo_label = emo_label[emo_mask]
            emo_loss = self.emo_criterion(emo_output, emo_label)

            spk_output = spk_output[spk_mask]
            spk_label = spk_label[spk_mask]
            spk_loss = self.spk_criterion(spk_output, spk_label)

            loss = self.multi_loss(emo_loss, spk_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

            self.optimizer.step()
            self.model.zero_grad()

            loss_array.append(loss.item())

            emo_pred = torch.argmax(emo_output, -1)
            emo_pred_array.append(emo_pred.cpu().numpy())
            emo_gold_array.append(emo_label.cpu().numpy())

        self.scheduler.step()
        emo_gold_array = np.concatenate(emo_gold_array)
        emo_pred_array = np.concatenate(emo_pred_array)

        f1 = f1_score(emo_gold_array, emo_pred_array, average='weighted')

        loss = np.mean(loss_array)

        return loss, f1

    def eval(self, data_loader):
        self.model.eval()
        loss_array = []
        emo_gold_array = []
        emo_pred_array = []
        with torch.no_grad():
            for dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask in data_loader:
                dia_input = dia_input.cuda()
                emo_label = emo_label.cuda()
                spk_label = spk_label.cuda()
                cls_index = cls_index.cuda()
                emo_mask = emo_mask.cuda()
                spk_mask = spk_mask.cuda()

                emo_output, spk_output = self.model(dia_input, cls_index, emo_mask)

                emo_output = emo_output[emo_mask]
                emo_label = emo_label[emo_mask]
                emo_loss = self.emo_criterion(emo_output, emo_label)

                spk_output = spk_output[spk_mask]
                spk_label = spk_label[spk_mask]
                spk_loss = self.spk_criterion(spk_output, spk_label)

                loss = self.multi_loss(emo_loss, spk_loss)

                loss_array.append(loss.item())

                emo_pred = torch.argmax(emo_output, -1)
                emo_pred_array.append(emo_pred.cpu().numpy())
                emo_gold_array.append(emo_label.cpu().numpy())

        emo_gold_array = np.concatenate(emo_gold_array)
        emo_pred_array = np.concatenate(emo_pred_array)

        f1 = f1_score(emo_gold_array, emo_pred_array, average='weighted')
        loss = np.mean(loss_array)

        return loss, f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_model', type=int, default=768, help='the max sequence length')
    parser.add_argument('--d_ff', type=int, default=768, help='the max sequence length')
    parser.add_argument('--heads', type=int, default=6, help='the hidden size of bert')
    parser.add_argument('--layers', type=int, default=1, help='the hidden size of bert')
    parser.add_argument('--input_max_length', type=int, default=512, help='the max sequence length')
    parser.add_argument('--hidden_dim', type=int, default=768, help='the hidden size of bert')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    logger = utils.get_logger("./log/HiTrans_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
    logger.info(args)

    torch.cuda.set_device(args.device)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logger.info("Loading data...")

    (train_set, dev_set, test_set), vocab = dataset.load_data(args.input_max_length)
    if args.evaluate:
        dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn,
                                  shuffle=True)
        dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        # test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    model = HiTrans(args.hidden_dim,
                    len(vocab.label2id),
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    heads=args.heads,
                    layers=args.layers,
                    dropout=args.dropout)

    trainer = Trainer(model)

    if args.evaluate:
        trainer.load("./checkpoint/model.pkl")
        dev_loss, dev_f1 = trainer.eval(dev_loader)
        logger.info("Dev Loss: {:.4f} F1: {:.4f}".format(dev_loss, dev_f1))
        test_loss, test_f1 = trainer.eval(test_loader)
        logger.info("Test Loss: {:.4f} F1: {:.4f}".format(test_loss, test_f1))
    else:
        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss, train_f1 = trainer.train(train_loader)
            logger.info("Epoch: {} Train Loss: {:.4f} F1: {:.4f}".format(epoch, train_loss, train_f1))
            dev_loss, dev_f1 = trainer.eval(dev_loader)
            logger.info("Epoch: {} Dev Loss: {:.4f} F1: {:.4f}".format(epoch, dev_loss, dev_f1))
            # test_loss, test_f1 = trainer.eval(test_loader)
            # logger.info("Test Loss: {:.4f} F1: {:.4f}".format(test_loss, test_f1))
            logger.info("---------------------------------")
            if best_f1 < dev_f1:
                best_f1 = dev_f1
                trainer.save("./checkpoint/model.pkl")

        logger.info("Best Dev F1: {:.4f}".format(best_f1))
