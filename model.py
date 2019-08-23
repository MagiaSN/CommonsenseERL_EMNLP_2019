import torch
import torch.nn as nn


class TensorComposition(nn.Module):
    def __init__(self, k, n1, n2):
        super(TensorComposition, self).__init__()
        self.k = k
        self.n1 = n1
        self.n2 = n2
        self.t = nn.Parameter(torch.FloatTensor(k, n1, n2))
        # torch.nn.init.xavier_uniform_(self.t, gain=1)
        torch.nn.init.normal_(self.t, std=0.01)

    def forward(self, a, b):
        '''
        a: (*, n1)
        b: (*, n2)
        '''
        k = self.k
        n1 = self.n1
        n2 = self.n2
        output_shape = tuple(a.size()[:-1] + (k,))  # (*, k)
        a = a.view(-1, n1)      # (m, n1)
        b = b.view(-1, n2)      # (m, n2)
        o = torch.einsum('ijk,cj,ck->ci', [self.t, a, b])
        return o.view(output_shape)


class LowRankTensorComposition(nn.Module):
    def __init__(self, k, n, r):
        super(LowRankTensorComposition, self).__init__()
        self.k = k
        self.n = n
        self.r = r
        self.t_l = nn.Parameter(torch.FloatTensor(k, n, r))
        self.t_r = nn.Parameter(torch.FloatTensor(k, r, n))
        self.t_diag = nn.Parameter(torch.FloatTensor(k, n))
        torch.nn.init.xavier_uniform_(self.t_l, gain=1)
        torch.nn.init.xavier_uniform_(self.t_r, gain=1)
        torch.nn.init.xavier_uniform_(self.t_diag, gain=1)

    def forward(self, a, b):
        '''
        a: (*, n)
        b: (*, n)
        '''
        k = self.k
        n = self.n
        output_shape = tuple(a.size()[:-1]) + (k,)
        # make t_diag
        t_diag = []
        for v in self.t_diag:
            t_diag.append(torch.diag(v))    # (n, n)
        t_diag = torch.stack(t_diag)        # (k, n, n)
        t = torch.bmm(self.t_l, self.t_r) + t_diag      # (k, n, n)
        a = a.view(-1, n)           # (m, n)
        b = b.view(-1, n)           # (m, n)
        o = torch.einsum('ijk,cj,ck->ci', [t, a, b])
        return o.view(output_shape)


class NeuralTensorNetwork(nn.Module):
    def __init__(self, embeddings, k):
        super(NeuralTensorNetwork, self).__init__()
        self.embeddings = embeddings
        self.vocab_size, self.emb_dim = embeddings.weight.size()
        self.k = k
        self.subj_verb_comp = TensorComposition(k, self.emb_dim, self.emb_dim)
        self.verb_obj_comp = TensorComposition(k, self.emb_dim, self.emb_dim)
        self.final_comp = TensorComposition(k, k, k)
        self.linear1 = nn.Linear(2 * self.emb_dim, k)
        self.linear2 = nn.Linear(2 * self.emb_dim, k)
        self.linear3 = nn.Linear(2 * k, k)
        self.tanh = nn.Tanh()

    def forward(self, subj_id, subj_w, verb_id, verb_w, obj_id, obj_w):
        '''
        subj_id: (batch, n)
        subj_w:  (batch, n)
        verb_id: (batch, n)
        verb_w:  (batch, n)
        obj_id:  (batch, n)
        obj_w:   (batch, n)
        '''
        batch_size = subj_id.size(0)
        emb_dim = self.emb_dim
        k = self.k
        # subject embedding
        subj_w = subj_w.unsqueeze(1)            # (batch, 1, n)
        subj = self.embeddings(subj_id)         # (batch, n, emb_dim)
        subj = torch.bmm(subj_w, subj)          # (batch, 1, emb_dim)
        subj = subj.squeeze(1)                  # (batch, emb_dim)
        # verb embedding
        verb_w = verb_w.unsqueeze(1)            # (batch, 1, n)
        verb = self.embeddings(verb_id)         # (batch, n, emb_dim)
        verb = torch.bmm(verb_w, verb)          # (batch, 1, emb_dim)
        verb = verb.squeeze(1)                  # (batch, emb_dim)
        # obj embedding
        obj_w = obj_w.unsqueeze(1)              # (batch, 1, n)
        obj = self.embeddings(obj_id)           # (batch, n, emb_dim)
        obj = torch.bmm(obj_w, obj)             # (batch, 1, emb_dim)
        obj = obj.squeeze(1)                    # (batch, emb_dim)
        # r1 = subj_verb_comp(subj, verb)
        tensor_comp = self.subj_verb_comp(subj, verb)   # (batch, k)
        cat = torch.cat((subj, verb), dim=1)    # (batch, 2*emb_dim)
        linear = self.linear1(cat)              # (batch, k)
        r1 = self.tanh(tensor_comp + linear)    # (batch, k)
        # r2 = verb_obj_comp(verb, obj)
        tensor_comp = self.verb_obj_comp(verb, obj)     # (batch, k)
        cat = torch.cat((verb, obj), dim=1)     # (batch, 2*emb_dim)
        linear = self.linear2(cat)              # (batch, k)
        r2 = self.tanh(tensor_comp + linear)    # (batch, k)
        # r3 = final_comp(r1, r2)
        tensor_comp = self.final_comp(r1, r2)   # (batch, k)
        cat = torch.cat((r1, r2), dim=1)        # (batch, 2*k)
        linear = self.linear3(cat)              # (batch, k)
        r3 = self.tanh(tensor_comp + linear)    # (batch, k)
        return r3


class LowRankNeuralTensorNetwork(NeuralTensorNetwork):
    def __init__(self, embeddings, k, r):
        super(NeuralTensorNetwork, self).__init__()
        self.embeddings = embeddings
        self.vocab_size, self.emb_dim = embeddings.weight.size()
        self.k = k
        self.r = r
        self.subj_verb_comp = LowRankTensorComposition(k, self.emb_dim, r)
        self.verb_obj_comp = LowRankTensorComposition(k, self.emb_dim, r)
        self.final_comp = LowRankTensorComposition(k, k, r)
        self.linear1 = nn.Linear(2 * self.emb_dim, k)
        self.linear2 = nn.Linear(2 * self.emb_dim, k)
        self.linear3 = nn.Linear(2 * k, k)
        self.tanh = nn.Tanh()


class BiLSTMEncoder(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers):
        super(BiLSTMEncoder, self).__init__()
        self.embeddings = embeddings
        vocab_size, emb_dim = embeddings.weight.size()
        self.bi_lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, input):
        '''
        input: (batch, n)
        '''
        embedding = self.embeddings(input)  # (batch, n, emb_dim)
        hidden, _ = self.bi_lstm(embedding) # (batch, n, 2 * hidden_size)
        hidden = hidden[:, -1, :]           # (batch, 2 * hidden_size)
        output = self.linear(hidden)        # (batch, hidden_size)
        return output        


class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, pos_score, neg_score):
        '''
        pos_score: (batch)
        neg_score: (batch)
        '''
        return torch.mean(self.relu(neg_score - pos_score + self.margin))


# deprecated

class EmbeddingWithBias(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(EmbeddingWithBias, self).__init__()
        self.emb_dim = emb_dim
        # use the last weight in dim 1 as bias
        self.embeddings = nn.Embedding(vocab_size, emb_dim + 1)

    def forward(self, input):
        emb = self.embeddings(input)
        shape = emb.size()
        w_shape = shape[:-1] + (-1,)
        b_shape = shape[:-1]
        emb = emb.view(-1, self.emb_dim + 1)
        w = emb[:,:-1]
        w = w.view(w_shape)
        b = emb[:,-1]
        b = b.view(b_shape)
        return w, b


# baselines

class Averaging(nn.Module):
    def __init__(self, embeddings):
        super(Averaging, self).__init__()
        self.embeddings = embeddings

    def forward(self, subj_id, subj_w, verb_id, verb_w, obj_id, obj_w):
        subj_emb = self.embeddings(subj_id)     # (batch, n, emb_dim)
        verb_emb = self.embeddings(verb_id)     # (batch, n, emb_dim)
        obj_emb = self.embeddings(obj_id)       # (batch, n, emb_dim)
        emb = torch.cat([subj_emb, verb_emb, obj_emb], dim=1)       # (batch, 3*n, emb_dim)
        subj_mask = subj_w.ne(0).float()        # (batch, n)
        verb_mask = verb_w.ne(0).float()        # (batch, n)
        obj_mask = obj_w.ne(0).float()          # (batch, n)
        mask = torch.cat([subj_mask, verb_mask, obj_mask], dim=1)   # (batch, 3*n)
        mask = mask.unsqueeze(1)                # (batch, 1, 3*n)
        output = torch.bmm(mask, emb)           # (batch, 1, emb_dim)
        output = output.squeeze(1)              # (batch, emb_dim)
        return output


class NN(nn.Module):
    def __init__(self, embeddings, hidden_size, output_size):
        super(NN, self).__init__()
        # e = W * tanh(H * [s:v:o])
        self.embeddings = embeddings
        vocab_size, emb_dim = embeddings.weight.data.size()
        self.h = nn.Linear(3 * emb_dim, hidden_size, bias=False)
        self.w = nn.Linear(hidden_size, output_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, subj_id, subj_w, verb_id, verb_w, obj_id, obj_w):
        # subject embedding
        subj_w = subj_w.unsqueeze(1)            # (batch, 1, n)
        subj = self.embeddings(subj_id)         # (batch, n, emb_dim)
        subj = torch.bmm(subj_w, subj)          # (batch, 1, emb_dim)
        subj = subj.squeeze(1)                  # (batch, emb_dim)
        # verb embedding
        verb_w = verb_w.unsqueeze(1)            # (batch, 1, n)
        verb = self.embeddings(verb_id)         # (batch, n, emb_dim)
        verb = torch.bmm(verb_w, verb)          # (batch, 1, emb_dim)
        verb = verb.squeeze(1)                  # (batch, emb_dim)
        # obj embedding
        obj_w = obj_w.unsqueeze(1)              # (batch, 1, n)
        obj = self.embeddings(obj_id)           # (batch, n, emb_dim)
        obj = torch.bmm(obj_w, obj)             # (batch, 1, emb_dim)
        obj = obj.squeeze(1)                    # (batch, emb_dim)
        # nn
        cat = torch.cat((subj, verb, obj), dim=1)   # (batch, 3*emb_dim)
        hidden = self.tanh(self.h(cat))             # (batch, hidden_size)
        output = self.w(hidden)                     # (batch, output_size)
        return output


class EMC(nn.Module):
    def __init__(self, embeddings, hidden_size, output_size):
        super(EMC, self).__init__()
        # e = W * tanh(H * [s:v:o:vs:vo])
        self.embeddings = embeddings
        vocab_size, emb_dim = embeddings.weight.data.size()
        self.h = nn.Linear(5 * emb_dim, hidden_size, bias=False)
        self.w = nn.Linear(hidden_size, output_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, subj_id, subj_w, verb_id, verb_w, obj_id, obj_w):
        # subject embedding
        subj_w = subj_w.unsqueeze(1)            # (batch, 1, n)
        subj = self.embeddings(subj_id)         # (batch, n, emb_dim)
        subj = torch.bmm(subj_w, subj)          # (batch, 1, emb_dim)
        subj = subj.squeeze(1)                  # (batch, emb_dim)
        # verb embedding
        verb_w = verb_w.unsqueeze(1)            # (batch, 1, n)
        verb = self.embeddings(verb_id)         # (batch, n, emb_dim)
        verb = torch.bmm(verb_w, verb)          # (batch, 1, emb_dim)
        verb = verb.squeeze(1)                  # (batch, emb_dim)
        # obj embedding
        obj_w = obj_w.unsqueeze(1)              # (batch, 1, n)
        obj = self.embeddings(obj_id)           # (batch, n, emb_dim)
        obj = torch.bmm(obj_w, obj)             # (batch, 1, emb_dim)
        obj = obj.squeeze(1)                    # (batch, emb_dim)
        # emc
        vs = verb * subj
        vo = verb * obj
        cat = torch.cat((subj, verb, obj, vs, vo), dim=1)   # (batch, 5*emb_dim)
        hidden = self.tanh(self.h(cat))                     # (batch, hidden_size)
        output = self.w(hidden)                             # (batch, output_size)
        return output


class PredicateTensorModel(nn.Module):
    def __init__(self, embeddings):
        super(PredicateTensorModel, self).__init__()
        self.embeddings = embeddings
        vocab_size, emb_dim = embeddings.weight.data.size()
        self.w = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim, emb_dim))
        self.u = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, std=0.01)
        torch.nn.init.normal_(self.u, std=0.01)

    def forward(self, subj_id, subj_w, verb_id, verb_w, obj_id, obj_w):
        '''
        subj_id: (batch, n)
        subj_w:  (batch, n)
        verb_id: (batch, n)
        verb_w:  (batch, n)
        obj_id:  (batch, n)
        obj_w:   (batch, n)
        '''
        # subj
        subj_w = subj_w.unsqueeze(1)            # (batch, 1, n)
        subj = self.embeddings(subj_id)         # (batch, n, emb_dim)
        subj = torch.bmm(subj_w, subj)          # (batch, 1, emb_dim)
        subj = subj.squeeze(1)                  # (batch, emb_dim)
        # verb
        verb_w = verb_w.unsqueeze(1)            # (batch, 1, n)
        verb = self.embeddings(verb_id)         # (batch, n, emb_dim)
        verb = torch.bmm(verb_w, verb)          # (batch, 1, emb_dim)
        verb = verb.squeeze(1)                  # (batch, emb_dim)
        # obj
        obj_w = obj_w.unsqueeze(1)              # (batch, 1, n)
        obj = self.embeddings(obj_id)           # (batch, n, emb_dim)
        obj = torch.bmm(obj_w, obj)             # (batch, 1, emb_dim)
        obj = obj.squeeze(1)                    # (batch, emb_dim)
        # predicate tensor composition
        alpha = torch.einsum('ijk,ci->jkc', [self.w, verb])     # (emb_dim, emb_dim, batch_size)
        obs = torch.stack([torch.diag(x) for x in obj], dim=2)  # (emb_dim, emb_dim, batch_size)
        gamma = torch.einsum('iak,ajk->ijk', [obs, alpha])      # (emb_dim, emb_dim, batch_size)
        l = torch.einsum('ijk,jkc->ikc', [self.u, gamma])       # (emb_dim, emb_dim, batch_size)
        final = torch.einsum('jic,cj->ci', [l, subj])           # (batch_size, emb_dim)
        return final


class RoleFactoredTensorModel(nn.Module):
    def __init__(self, embeddings, k):
        super(RoleFactoredTensorModel, self).__init__()
        self.embeddings = embeddings
        self.vocab_size, self.emb_dim = embeddings.weight.size()
        self.k = k
        self.tensor_comp = TensorComposition(k, self.emb_dim, self.emb_dim)
        self.w = nn.Linear(2 * k, k, bias=False)

    def forward(self, subj_id, subj_w, verb_id, verb_w, obj_id, obj_w):
        '''
        subj_id: (batch, n)
        subj_w:  (batch, n)
        verb_id: (batch, n)
        verb_w:  (batch, n)
        obj_id:  (batch, n)
        obj_w:   (batch, n)
        '''
        emb_dim = self.emb_dim
        k = self.k
        batch_size = subj_id.size(0)
        # subj
        subj_w = subj_w.unsqueeze(1)            # (batch, 1, n)
        subj = self.embeddings(subj_id)         # (batch, n, emb_dim)
        subj = torch.bmm(subj_w, subj)          # (batch, 1, emb_dim)
        subj = subj.squeeze(1)                  # (batch, emb_dim)
        # verb
        verb_w = verb_w.unsqueeze(1)            # (batch, 1, n)
        verb = self.embeddings(verb_id)         # (batch, n, emb_dim)
        verb = torch.bmm(verb_w, verb)          # (batch, 1, emb_dim)
        verb = verb.squeeze(1)                  # (batch, emb_dim)
        # obj
        obj_w = obj_w.unsqueeze(1)              # (batch, 1, n)
        obj = self.embeddings(obj_id)           # (batch, n, emb_dim)
        obj = torch.bmm(obj_w, obj)             # (batch, 1, emb_dim)
        obj = obj.squeeze(1)                    # (batch, emb_dim)
        # vs, vo
        vs = self.tensor_comp(verb, subj)       # (batch, k)
        vo = self.tensor_comp(verb, obj)        # (batch, k)
        cat = torch.cat((vs, vo), dim=1)        # (batch, 2*k)
        return self.w(cat)                      # (batch, k)
