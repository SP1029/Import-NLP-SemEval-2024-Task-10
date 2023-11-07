import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers import *
from utils import *

NUM_OF_EMOTIONS = 7

class ERC_MMN(nn.Module):
    def __init__(self,hidden_size,weight_matrix,utt2idx,batch_size,seq_len):
        super(ERC_MMN,self).__init__()
        self.ia = Interact(hidden_size,weight_matrix,utt2idx)
        self.mn = MemoryNetwork(4,hidden_size,batch_size,seq_len)
        self.pool = Pool()
        
        self.rnn_c = RNN_modified(hidden_size*3, hidden_size*2, 1) # num_layers = 1
        
        self.rnn_e = RNN_modified(hidden_size*2, hidden_size*2, 1)
                
        self.linear1 = dense_E(hidden_size*2, NUM_OF_EMOTIONS) # number of emotions are 7

    def forward(self,c_ids,speaker_info,sp_dialogues,sp_em,sp_ind,x1,mode="train"):
        glob, splvl = self.ia(c_ids,speaker_info,sp_dialogues,sp_ind,x1)

        op = self.mn(glob,splvl)
        op = self.pool(op)

        op = torch.cat([splvl,op],dim=2)

        rnn_c_op,_ = self.rnn_c(op)

        rnn_e_op,_ = self.rnn_e(rnn_c_op)
        fip = rnn_e_op.add(rnn_c_op)      # Residual Connection
        fop1 = self.linear1(fip)

        return fip,fop1