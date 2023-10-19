import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *

class RNN_modified(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout=0.1,
            bidirectional=False,
    ):
        super(RNN_modified, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.RNN = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
    
    def forward(self, x, h0=None, c0=None):
        output, (hidden, cell) = self.RNN(x, (h0, c0))
        return output, hidden, cell
    
class Attention(nn.Module):
    def __init__(
            self,
            Q_dim,
            K_dim=None,
            V_dim=None,
            hidden_dim=None,
            O_dim=None,
            dropout=0.1,
    ):
        super(Attention, self).__init__()
        self.Q_dim = Q_dim
        if K_dim is None:
            self.K_dim = Q_dim
        else:
            self.K_dim = K_dim
        if V_dim is None:
            self.V_dim = Q_dim
        else:
            self.V_dim = V_dim
        if hidden_dim is None:
            self.hidden_dim = Q_dim
        else:
            self.hidden_dim = hidden_dim
        if O_dim is None:
            self.O_dim = Q_dim
        else:
            self.O_dim = O_dim

        self.keys = nn.Linear(self.K_dim, self.hidden_dim)
        self.queries = nn.Linear(self.Q_dim, self.hidden_dim)
        self.values = nn.Linear(self.V_dim, self.O_dim)
        self.dropout = nn.Dropout(dropout)

    def mask_score(self, score, mask):
        mask = mask.unsqueeze(1)
        score = score.masked_fill(mask == 0, -1e9)
        return score
    
    def forward(self, key, query, mask=None):
        if len(query.shape) == 1:
            query = query.unsqueeze(0)
        if len(key.shape) == 1:
            key = key.unsqueeze(0)

        if len(query.shape) == 2:
            query = query.unsqueeze(1)  
        if len(key.shape) == 2:
            key = key.unsqueeze(1)

        query = self.queries(query)
        key = self.keys(key)
        value = self.values(key)

        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        if mask != None:
            score = self.mask_score(score, mask)

        score = F.softmax(score, dim=-1)
        score = self.dropout(score)

        score.data[score != score] = 0  # remove nan from softmax operation
        output = torch.bmm(score, value)
        return output, score
    
class Interact(nn.Module):
    def __init__(self, hidden_dim, weight_matrix, utt2idx, dropout=0.1):
        super(Interact, self).__init__()
        self.hidden_size = hidden_dim
        self.embedding, num_embed, embedding_dim = create_emb_layer(weight_matrix, utt2idx)

        # Dialogue RNN
        self.RNN_D = RNN_modified(embedding_dim, hidden_dim, 1, bidirectional=False)
        self.dropout_1 = nn.Dropout(dropout)

        # TODO: number of layers to be increased
        self.RNN_G = RNN_modified(embedding_dim * 3, hidden_dim, 1)
        self.dropout_2 = nn.Dropout(dropout)    

        self.attention = Attention(embedding_dim)

        self.RNN_S = RNN_modified(embedding_dim * 2, embedding_dim * 2, 1)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, chat_ids, speaker_info, sp_dialogue, sp_ind, inputs):
        bert_embeddings = self.embedding(inputs)
        dialogue, _, _ = self.RNN_D(bert_embeddings)    # get global representation of dialogue
        dialogue = self.dropout_1(dialogue)

        self.device = inputs.device

        # fop: prev. state of out
        # spop: speaker level output from RNN_S
        # fop2: concatenation of the speaker-specific 
        #       tensor spop and the global dialogue tensor
        #       effectively combining speaker and dialogue 
        #       information
        # op: is the output of the global RNN (RNN_G)
        #     after processing the 
        #     concatenated tensor fop2

        fop = torch.zeros((dialogue.size()[0], dialogue.size()[1], dialogue.size()[2])).to(self.device)
        fop2 = torch.zeros((dialogue.size()[0], dialogue.size()[1], dialogue.size()[2]*3)).to(self.device)
        op = torch.zeros((dialogue.size()[0], dialogue.size()[1], dialogue.size()[2])).to(self.device)
        spop = torch.zeros((dialogue.size()[0], dialogue.size()[1], dialogue.size()[2]*2)).to(self.device)

        h_0 = torch.randn(1, 1, self.hidden_size * 2).to(self.device)
        d_h = torch.randn(1, 1, self.hidden_size).to(self.device)
        attention_h = torch.randn(1, 1, self.hidden_size).to(self.device)

        for i in range(dialogue.size()[0]):
            d_id = chat_ids[i]
            speaker_hidden_states = {}
            for diag in range(dialogue.size()[1]):
                fop = op.clone()

                current_utt = dialogue[i][diag]
                current_speaker = speaker_info[d_id][diag]

                if current_speaker not in speaker_hidden_states:
                    speaker_hidden_states[current_speaker] = h_0

                h = speaker_hidden_states[current_speaker]
                current_utt_emb = torch.unsqueeze(torch.unsqueeze(current_utt, 0), 0)

                key = torch.unsqueeze(fop[i][:diag + 1].clone(), 0)

                # a change in the model architecture (averaging the cell states)
                avg_cell = 0

                if diag == 0:
                    spop[i][diag], h_new, cell = self.RNN_S(
                        torch.cat([attention_h, current_utt_emb], -1).to(self.device), 
                        h, 
                        None
                    )
                    avg_cell = cell
                else:
                    attention_op, _ = self.attention(key, current_utt_emb) # query = current_utt_emb, key = fop[i][:s + 1].clone()
                    spop[i][diag], h_new, cell = self.RNN_S(
                        torch.cat([attention_op, current_utt_emb], -1).to(self.device), 
                        h, 
                        cell
                    )
                    avg_cell += cell

                avg_cell /= (diag + 1)  # averaging the cell states

                # Residual Connection
                spop[i][diag] = spop[i][diag].add(torch.cat([attention_h, current_utt_emb], -1).to(self.device))
                speaker_hidden_states[current_speaker] = h_new
                fop2[i][diag] = torch.cat([spop[i][diag], fop[i][diag]], -1).to(self.device) # check: to(self.device)

                # using the average cell state
                op[i][diag], _, _ = self.RNN_G(fop2[i][diag].clone(), d_h, avg_cell)

        return op, spop

class dense_E(nn.Module):
    def __init__(self, input_dim, output_dim, dropout1=0.1, dropout2=0.6, dropout3=0.7):
        super(dense_E, self).__init__()

        # TODO: modify the architecture

        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.linear2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.dropout2 = nn.Dropout(dropout2)

        self.linear3 = nn.Linear(input_dim // 4, output_dim)
        self.dropout3 = nn.Dropout(dropout3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout2(F.relu(self.linear2(x)))
        x = self.dropout3(F.relu(self.linear3(x)))
        return x
    
# class dense_T(nn.Module):[DON'T ACTUALLY NEED THIS, probably redundant]

class MaskedAttention(nn.Module):
    def __init__(self, batch_size, seq_len, embedding_size):
        super(MaskedAttention, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.attention = Attention(
            Q_dim=embedding_size,
            K_dim=embedding_size,
            O_dim=embedding_size,
        )

    def create_mask(self, n):
        mask = torch.zerors((1, self.seq_len, self.embedding_size), dtype=torch.float32)
        mask[:, n + 1] = torch.ones((1, self.embedding_size), dtype=torch.float32).repeat(self.batch_size, 1, 1)
        return mask
    
    def forward(self, key, query):
        self.device = key.device

        outputs = torch.zeros_like(key).astype(torch.float32).to(self.device)
        for i in range(key.size()[1]):
            mask = self.create_mask(i)
            output, _ = self.attention(key, query, mask)
            for b in range(output.size()[0]):
                outputs[b][i] = output[b][i] 
        return outputs
    
class MemoryNetwork(nn.Module):
    def __init__(
            self,
            num_hops,   # hopping refers to the 
                        # repeated application of
                        # attention mechanisms over
                        # the memory slots 
                        # or sequences to 
                        # refine the output
            hidden_dim,
            batch_size,
            seq_len,
    ):
        self.num_hops = num_hops
        self.RNN = RNN_modified(hidden_dim, hidden_dim, 1)
        self.masked_attn = MaskedAttention(batch_size, seq_len, hidden_dim)

    def forward(self, globl, spl):
        X = globl
        for i in range(self.num_hops):
            diag, _, _ = self.RNN()
            X = self.RNN(X)
            X = self.masked_attn(diag, spl)
        return X
    
class Pool(nn.Module):
    def __init__(self, mode="mean"):
        super(Pool, self).__init__()
        self.mode = mode
    
    def forward(self, x):
        self.device = x.device
        output = torch.zeros_like(x).to(self.device)
        for b in range(x.size()[0]):
            this_tensor = []
            for s in range(x.size()[1]):
                this_tensor.append(x[b][s])
                if self.mode == "mean":
                    output[b][s] = torch.mean(torch.stack(this_tensor), 0)
                elif self.mode == "max":
                    output[b][s] = torch.max(torch.stack(this_tensor), 0)
                elif self.mode == "sum":
                    output[b][s] = torch.sum(torch.stack(this_tensor), 0)
                else:
                    raise NotImplementedError
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()        
        self.dropout = nn.Dropout(p=dropout)        
        pe = torch.zeros(max_len, d_model)        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))        
        pe[:, 0::2] = torch.sin(position * div_term)        
        pe[:, 1::2] = torch.cos(position * div_term)        
        pe = pe.unsqueeze(0).transpose(0, 1)        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]        
        return self.dropout(x)



