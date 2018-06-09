import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import to_var


class CharEmbedding(nn.Module):
    '''
     In : (N, sentence_len, word_len, vocab_size_c)
     Out: (N, sentence_len, c_embd_size)
     '''
    def __init__(self, args):
        super(CharEmbedding, self).__init__()
        self.embd_size = args.c_embd_size
        self.embedding = nn.Embedding(args.vocab_size_c, args.c_embd_size)
        self.conv = nn.ModuleList([nn.Conv2d(1, args.out_chs, (f[0], f[1])) for f in args.filters])
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        # x: (N, seq_len, word_len)
        input_shape = x.size()
        word_len = x.size(2)
        
        #flatten the data in order to pass to embedding layer
        x = x.view(-1, word_len) # (N*seq_len, word_len)
        x = self.embedding(x) # (N*seq_len, word_len, c_embd_size)
        
        #reshape back
        x = x.view(*input_shape, -1) # (N, seq_len, word_len, c_embd_size)
        x = x.sum(2) # (N, seq_len, c_embd_size)

        # CNN
        x = x.unsqueeze(1) # (N, Cin, seq_len, c_embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (N,Cin, Hin, Win )
        #    Output: (N,Cout,Hout,Wout)
        x = [F.relu(conv(x)) for conv in self.conv] # (N, Cout, seq_len, c_embd_size-filter_w+1). stride == 1
        # [(N,Cout,Hout,Wout) -> [(N,Cout,Hout*Wout)] * len(filter_heights)
        # [(N, seq_len, c_embd_size-filter_w+1, Cout)] * len(filter_heights)
        x = [xx.view((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]
        # maxpool like
        # [(N, seq_len, Cout)] * len(filter_heights)
        x = [torch.sum(xx, 2) for xx in x]
        # (N, seq_len, Cout==word_embd_size)
        x = torch.cat(x, 1)
        x = self.dropout(x)

        return x

class WordEmbedding(nn.Module):

    def __init__(self, args, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size_w, args.w_embd_size)
        if args.pre_embd_w is not None:
            self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)


class Highway(nn.Module):
    def __init__(self, in_size, n_layers=2, act=F.relu):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.act = act

        self.normal_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            normal_layer_ret = self.act(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x

class GRU_With_Dropout(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=True, batch_first=True):
        super(GRU_With_Dropout, self).__init__()
        #Pytorch GRU implementation apply dropout to only output
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)
        
    def forward(self, x, hidden=None):
        gru_out, _h = self.gru(x)
        return self.dropout(gru_out), _h

class ContextualEmbedding(nn.Module):
    def __init__(self, args):
        super(ContextualEmbedding, self).__init__()
        self.d = args.w_embd_size*2
        self.char_embd_net = CharEmbedding(args)
        self.word_embd_net = WordEmbedding(args)
        self.highway_net = Highway(self.d)
        self.ctx_embd_layer = GRU_With_Dropout(self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True)
    
    def forward(self, char, word):
        # 1. Character Embedding Layer
        char_embd = self.char_embd_net(char) # (N, seq_len, embd_size)
        # 2. Word Embedding Layer
        word_embd = self.word_embd_net(word) # (N, seq_len, embd_size)
        # Highway Networks for 1. and 2.
        embd = torch.cat((char_embd, word_embd), 2) # (N, seq_len, d=embd_size*2)
        embd = self.highway_net(embd) # (N, seq_len, d=embd_size*2)

        # 3. Contextual  Embedding Layer
        ctx_embd_out, _h = self.ctx_embd_layer(embd)
        return ctx_embd_out


class AttentionalFlow(nn.Module):
    def __init__(self, args):
        super(AttentionalFlow, self).__init__()
        self.d = args.w_embd_size*2
        self.W = nn.Linear(6*self.d, 1, bias=False)
    
    def forward(self, embd_context, embd_query):
        batch_size = embd_context.size(0)
        T = embd_context.size(1)   # context sentence length (word level)
        J = embd_query.size(1)     # query sentence length   (word level)
        
         # 4. Attention Flow Layer
        # Make a similarity matrix
        shape = (batch_size, T, J, 2*self.d)            # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2)     # (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape) # (N, T, J, 2d)
        embd_query_ex = embd_query.unsqueeze(1)         # (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)     # (N, T, J, 2d)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex) # (N, T, J, 2d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3) # (N, T, J, 6d), [h;u;h◦u]
        S = self.W(cat_data).view(batch_size, T, J) # (N, T, J)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), embd_query) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1) # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), embd_context) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1) # (N, T, 2d), tiled T times

        # G: query aware representation of each context word
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2) # (N, T, 8d)
        return G

class AnswerLayer(nn.Module):
    def __init__(self, args):
        super(AnswerLayer, self).__init__()
        self.d = args.w_embd_size*2
        self.bilinear_w1 = nn.Linear(2*self.d, 2*self.d, bias=False)
        self.bilinear_w2 = nn.Linear(2*self.d, 4*self.d, bias=False)
        self.bilinear_beta = nn.Linear(2*self.d, 2*self.d, bias=False)
        self.next_state_lstm = nn.GRU(2*self.d, 2*self.d, dropout=0, bidirectional=False, batch_first=True)
    def forward(self, state, M):
        # state: (N, 1, 2d)
        # M: (N, T, 2d)
        T = M.size(1)
        
        logits1 = torch.bmm(self.bilinear_w1(M), state.squeeze().unsqueeze(-1)).squeeze() #(N, T, 2d)*(2d, 2d)*(N, 2d, 1)

        p1 = F.softmax(logits1, dim=-1) # (N, T)
  
        p1mem = torch.bmm(p1.unsqueeze(1), M) # (N, 1, 2d)
        p1s = torch.cat((p1mem, state), -1) # (N, 1, 4d)
        
        logits2 = torch.bmm(self.bilinear_w2(M), p1s.squeeze().unsqueeze(-1)).squeeze() #(N, T, 2d)*(2d, 4d)*(N, 4d, 1)

        p2 = F.softmax(logits2, dim=-1) # (N, T)
    
        logits_beta = torch.bmm(self.bilinear_beta(M), state.squeeze().unsqueeze(-1)) #(N, T, 2d)*(2d, 2d)*(N, 2d, 1)
        beta = F.softmax(logits_beta, dim=-1).view((-1, 1, T)) # (N, 1, T)
        x_t = torch.bmm(beta, M) # (N, 1, 2d)

        state_2, _h = self.next_state_lstm(state, x_t.squeeze().unsqueeze(0)) #(N, 1, 2d)
        return state_2, p1, p2

class BiDAF(nn.Module):
    def __init__(self, args):
        super(BiDAF, self).__init__()
        self.d = args.w_embd_size*2
        
        self.ctx_embd_layer = ContextualEmbedding(args)
        self.attentional_flow_layer = AttentionalFlow(args)

        self.modeling_layer = GRU_With_Dropout(8*self.d, self.d, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True)


        self.p1_layer = nn.Sequential(nn.Dropout(0.2), nn.Linear(10*self.d, 1, bias=False))
        self.p2_lstm_layer = GRU_With_Dropout(14*self.d, self.d, dropout=0.2, bidirectional=True, batch_first=True)

        self.p2_layer = nn.Sequential(nn.Dropout(0.2), nn.Linear(10*self.d, 1))
        
        self.self_attn = nn.Linear(2*self.d, 1, bias=False)

        self.answer_layer = AnswerLayer(args)
    def forward(self, ctx_w, ctx_c, query_w, query_c):


        # 1. Character Embedding Layer
        # 2. Word Embedding Layer
        # 3. Contextual  Embedding Layer
        embd_context = self.ctx_embd_layer(ctx_c, ctx_w) # (N, T, 2d)
        embd_query   = self.ctx_embd_layer(query_c, query_w) # (N, J, 2d)
        

        
        
        T = embd_context.size(1) 

        # 4. Attention Flow Layer
        G = self.attentional_flow_layer(embd_context, embd_query)
    
        # 5. Modeling Layer
        M, _h = self.modeling_layer(G) # M: (N, T, 2d)

        # 6. Output Layer

        # x: Self attention for embd query
        self_attention = F.softmax(self.self_attn(embd_query).squeeze(), dim=-1) # (N, J)
        state = torch.bmm(self_attention.unsqueeze(1), embd_query) #(N, 1, J) * (N, J, 2d) = (N, 1, 2d)

        p1_sum = to_var(torch.zeros(M.size(0), M.size(1)))
        p2_sum = to_var(torch.zeros(M.size(0), M.size(1)))
        p1_count = 1
        p2_count = 1
        
        _, p1, p2 = self.answer_layer(state, M) #p1, p2: (N, T)


        p1_sum += p1
        p2_sum += p2
        
#         for i in range(2):
#             state, p1_1, p2_1 = self.answer_layer(state, M) #p1, p2: (N, T)
#             if np.random.rand()>.4 or not self.training:
#                 p1_sum += p1_1
#                 p1_count += 1
#             if np.random.rand()>.4 or not self.training:
#                 p2_sum += p2_1
#                 p2_count += 1 

        return p1_sum/p1_count, p2_sum/p2_count