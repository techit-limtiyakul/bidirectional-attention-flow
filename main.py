import os
import shutil
from argparse import Namespace
import argparse
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import BiDAF
from dataset import load_processed_json, load_glove_weights, make_vector, DataSet


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=40, help='input batch size')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--w_embd_size', type=int, default=100, help='word embedding size')
parser.add_argument('--c_embd_size', type=int, default=8, help='character embedding size')
parser.add_argument('--start_epoch', type=int, default=0, help='resume epoch count, default=0')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--checkpoint_dir', default='./checkpoints/', type=str, metavar='PATH', help='path where trained parameters will be saved')
parser.add_argument('--data_dir', default='./data/squad/', type=str, metavar='PATH', help='dataset file directory')
parser.add_argument('--resume', default='./checkpoints/Epoch-12.model.test', type=str, metavar='PATH', help='file containing saved params')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
    
train_json, train_shared_json = load_processed_json('{}data_train.json'.format(args.data_dir), '{}shared_train.json'.format(args.data_dir))
test_json, test_shared_json = load_processed_json('{}data_test.json'.format(args.data_dir), '{}shared_test.json'.format(args.data_dir))
train_data = DataSet(train_json, train_shared_json)
test_data = DataSet(test_json, test_shared_json)
ctx_maxlen = train_data.get_ctx_maxlen()
ctx_sent_maxlen, query_sent_maxlen = train_data.get_sent_maxlen()
w2i_train, c2i_train = train_data.get_word_index()
w2i_test, c2i_test = test_data.get_word_index()
vocabs_w = sorted(list(set(list(w2i_train.keys()) + list(w2i_test.keys()))))
w2i = {w : i for i, w in enumerate(vocabs_w, 3)}
vocabs_c = sorted(list(set(list(c2i_train.keys()) + list(c2i_test.keys()))))
c2i = {c : i for i, c in enumerate(vocabs_c, 3)}

NULL = "-NULL-"
UNK = "-UNK-"
ENT = "-ENT-"

w2i[NULL] = 0
w2i[UNK] = 1
w2i[ENT] = 2
c2i[NULL] = 0
c2i[UNK] = 1
c2i[ENT] = 2


print('----')
print('n_train', train_data.size())
print('n_test', test_data.size())
print('ctx_maxlen', ctx_maxlen)
print('vocab_size_w:', len(w2i))
print('vocab_size_c:', len(c2i))
print('ctx_sent_maxlen:', ctx_sent_maxlen)
print('query_sent_maxlen:', query_sent_maxlen)

glove_embd_w = torch.from_numpy(load_glove_weights(args.data_dir, args.w_embd_size, len(w2i), w2i)).type(torch.FloatTensor)

args.vocab_size_c = len(c2i)
args.vocab_size_w = len(w2i)
args.pre_embd_w = glove_embd_w
args.filters = [[1, 5]]
args.out_chs = 100
args.ans_size = ctx_sent_maxlen


class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

def save_checkpoint(state, filename='./checkpoints/checkpoint.pth.tar'):
    print('model saved!', filename)
    torch.save(state, filename)

def custom_loss_fn(data, labels):
    loss = Variable(torch.zeros(1))
    for d, label in zip(data, labels):
        loss -= torch.log(d[label]).cpu()
    loss /= data.size(0)
    return loss


def train(model, data, optimizer, ema, n_epoch=30, start_epoch=0, batch_size=args.batch_size):
    print('----Train---')
    label = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model.train()
    for epoch in range(start_epoch, n_epoch):
        print('---Epoch', epoch)
        batches = data.get_batches(batch_size, shuffle=True)
        p1_acc, p2_acc = 0, 0
        total = 0
        for i, batch in enumerate(tqdm(batches)):
            # (c, cc, q, cq, a)
            ctx_sent_len   = max([len(d[0]) for d in batch])
            ctx_word_len   = max([len(w) for d in batch for w in d[1]])
            query_sent_len = max([len(d[2]) for d in batch])
            query_word_len = max([len(w) for d in batch for w in d[3]])
            c, cc, q, cq, ans_var = make_vector(batch, w2i, c2i, ctx_sent_len, ctx_word_len, query_sent_len, query_word_len)
            a_beg = ans_var[:, 0]
            a_end = ans_var[:, 1] - 1
            p1, p2 = model(c, cc, q, cq)

            loss_p1 = custom_loss_fn(p1, a_beg)
            loss_p2 = custom_loss_fn(p2, a_end)
            p1_acc += torch.sum(a_beg == torch.max(p1, 1)[1]).data[0]
            p2_acc += torch.sum(a_end == torch.max(p2, 1)[1]).data[0]
            total += len(batch)
            if (i+1) % 50 == 0:
                rep_str = '[{}] Epoch {} {:.1f}%, loss_p1: {:.3f}, loss_p2: {:.3f}'
                print(rep_str.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                     epoch,
                                     100*i/len(batches),
                                     loss_p1.data[0],
                                     loss_p2.data[0]))
                acc_str = 'p1 acc: {:.3f}% ({}/{}), p2 acc: {:.3f}% ({}/{})'
                print(acc_str.format(100*p1_acc/total,
                                     p1_acc,
                                     total,
                                     100*p2_acc/total,
                                     p2_acc,
                                     total))
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        offset = epoch * (len(batches) * batch_size)
                        step = i * batch_size + offset
                        name = name.replace('.', '/')

            optimizer.zero_grad()
            (loss_p1+loss_p2).backward()
            optimizer.step()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data = ema(name, param.data)

        print('======== Epoch {} result ========'.format(epoch))
        print('p1 acc: {:.3f}, p2 acc: {:.3f}'.format(100*p1_acc/total, 100*p2_acc/total))
        filename = '{}/Epoch-{}.model'.format(args.checkpoint_dir, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filename=filename)


# test() {{{
def test(model, data, batch_size=args.batch_size):
    print('----Test---')
    model.eval()
    p1_acc, p2_acc = 0, 0
    total = 0
    batches = data.get_batches(batch_size)
    for i, batch in enumerate(tqdm(batches)):
        # (c, cc, q, cq, a)
        ctx_sent_len   = max([len(d[0]) for d in batch])
        ctx_word_len   = max([len(w) for d in batch for w in d[1]])
        query_sent_len = max([len(d[2]) for d in batch])
        query_word_len = max([len(w) for d in batch for w in d[3]])
        c, cc, q, cq, ans_var = make_vector(batch, w2i, c2i, ctx_sent_len, ctx_word_len, query_sent_len, query_word_len)
        a_beg = ans_var[:, 0]
        a_end = ans_var[:, 1] - 1
        p1, p2 = model(c, cc, q, cq)
        p1_acc += torch.sum(a_beg == torch.max(p1, 1)[1]).data[0]
        p2_acc += torch.sum(a_end == torch.max(p2, 1)[1]).data[0]
        total += batch_size
        if i % 10 == 0:
            print('current acc: {:.3f}%'.format(100*p1_acc/total))

    print('======== Test result ========')
    print('p1 acc: {:.3f}%, p2 acc: {:.3f}%'.format(100*p1_acc/total, 100*p2_acc/total))
# }}}

#create model
model = BiDAF(args)

if torch.cuda.is_available():
    print('use cuda')
    model.cuda()


#resume
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))

ema = EMA(0.999)
for name, param in model.named_parameters():
    if param.requires_grad:
        ema.register(name, param.data)

print('parameters-----')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())

if args.test == 1:
    print('Test mode')
    test(model, test_data)
else:
    print('Train mode')
    train(model, train_data, optimizer, ema, start_epoch=args.start_epoch)
print('finish')

