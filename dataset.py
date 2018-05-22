import os
import numpy as np
import json
from nltk.tokenize import word_tokenize
import random
import torch
from torch.autograd import Variable

NULL = "-NULL-"
UNK = "-UNK-"
ENT = "-ENT-"

def lower_list(str_list):
    return [str_var.lower() for str_var in str_list]


def load_task(dataset_path):
    ret_data = []
    ctx_max_len = 0 # character level length
    with open(dataset_path) as f:
        data = json.load(f)
        ver = data['version']
        print('dataset version:', ver)
        data = data['data']
        for i, d in enumerate(data):
            if i % 100 == 0:
                print('load_task:', i, '/', len(data))
            for p in d['paragraphs']:
                if len(p['context']) > ctx_max_len:
                    ctx_max_len = len(p['context'])
                c = word_tokenize(p['context'])
                cc = [list(w) for w in c]
                q, a = [], []
                for qa in p['qas']:
                    q = word_tokenize(qa['question'])
                    qc = [list(w) for w in q]
                    a = [ans['text'] for ans in qa['answers']]
                    a_beg = [ans['answer_start'] for ans in qa['answers']]
                    a_end = [ans['answer_start'] + len(ans['text']) for ans in qa['answers']]
                    ret_data.append((c, cc, qa['id'], q, qc, a, a_beg, a_end)) 
    return ret_data, ctx_max_len


def load_processed_data(fpath):
    ctx_max_len = 0 # character level length
    with open(fpath) as f:
        lines = f.readlines()
        data = []
        for l in lines:
            c_label, c, q, a, a_txt = l.rstrip().split('\t')
            if len(c) > ctx_max_len:
                ctx_max_len = len(c)
            c, q, a = c.split(' '), q.split(' '), a.split(' ')
            c, q = lower_list(c), lower_list(q)
            cc = [list(w) for w in c]
            qc = [list(w) for w in q]
            a = [int(aa) for aa in a]
            a = [a[0], a[-1]]
            data.append((c_label, c, cc, q, qc, a, a_txt))
    return data, ctx_max_len


def load_processed_json(fpath_data, fpath_shared):
    data = json.load(open(fpath_data))
    shared = json.load(open(fpath_shared))
    return data, shared


def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.' + str(embd_dim) + 'd.txt'),encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector

    print('Found %s word vectors in glove.' % len(embeddings_index))
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_ct = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # words not found in embedding index will be all-zeros.
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_ct += 1
    print(found_ct, 'words are found in glove')

    return embedding_matrix


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def _make_word_vector(sentence, w2i, seq_len):
    index_vec = [w2i[w] if w in w2i else w2i[UNK] for w in sentence]
    pad_len = max(0, seq_len - len(index_vec))
    index_vec += [w2i[NULL]] * pad_len
    index_vec = index_vec[:seq_len]
    return index_vec


def _make_char_vector(data, c2i, sent_len, word_len):
    tmp = torch.ones(sent_len, word_len).type(torch.LongTensor) 
    for i, word in enumerate(data):
        for j, ch in enumerate(word):
            tmp[i][j] = c2i[ch] if ch in c2i else c2i[UNK]
    return tmp


def make_vector(batch, w2i, c2i, ctx_sent_len, ctx_word_len, query_sent_len, query_word_len):
    c, cc, q, cq, ans = [], [], [], [], []
    for d in batch:
        c.append(_make_word_vector(d[0], w2i, ctx_sent_len))
        cc.append(_make_char_vector(d[1], c2i, ctx_sent_len, ctx_word_len))
        q.append(_make_word_vector(d[2], w2i, query_sent_len))
        cq.append(_make_char_vector(d[3], c2i, query_sent_len, query_word_len))
        ans.append(d[-1])
    c = to_var(torch.LongTensor(c))
    cc = to_var(torch.stack(cc, 0))
    q = to_var(torch.LongTensor(q))
    cq = to_var(torch.stack(cq, 0))
    a = to_var(torch.LongTensor(ans))
    return c, cc, q, cq, a


class DataSet(object):
    def __init__(self, data, shared):
        self.data = data
        self.shared = shared

    def size(self):
        return len(self.data['q'])

    def get_batches(self, batch_size, shuffle=False):
        batches = []
        batch = []
        idxs = list(range(self.size()))
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(self.size()):
            rx = self.data['*x'][i] 
            c  = lower_list(self.shared['x'][rx[0]][rx[1]][0])
            # if len(c) > 150: continue
            cc = self.shared['cx'][rx[0]][rx[1]][0]
            q  = lower_list(self.data['q'][i])
            cq = self.data['cq'][i]
            a  = self.data['y'][i][0] 
            a  = (a[0][1], a[1][1])
            batch.append((c, cc, q, cq, a))
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
        return batches

    def get_ctx_maxlen(self):
        # char level context maxlen
        return max([len(p) for pp in self.shared['p'] for p in pp])

    def get_sent_maxlen(self):
        # word level sentence maxlen
        return max([len(articles[0]) for xx in self.shared['x'] for articles in xx]), max([len(q) for q in self.data['q']])

    def get_word_maxlen(self):
        # max word len
        return max([len(w) for xx in self.shared['x'] for articles in xx for w in articles[0]]), max([len(w) for q in self.data['q'] for w in q])

    def get_word_index(self, word_count_th=10, char_count_th=100):

        word2vec_dict = self.get_word2vec()
        word_counter = self.get_word_counter()
        char_counter = self.get_char_counter()
        w2i = {w: i for i, w in enumerate(w for w, ct in word_counter.items()
                                            if ct > word_count_th or (w in word2vec_dict))}
        c2i = {c: i for i, c in
                    enumerate(c for c, ct in char_counter.items()
                              if ct > char_count_th)}

        return w2i, c2i

    def get_word2vec(self):
        return self.shared['lower_word2vec']

    def get_word_counter(self):
        return self.shared['lower_word_counter']

    def get_char_counter(self):
        return self.shared['char_counter']