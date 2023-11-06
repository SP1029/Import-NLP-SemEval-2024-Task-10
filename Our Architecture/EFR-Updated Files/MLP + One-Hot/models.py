# MLP of 5 Layers 

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.i12i2 = torch.nn.Linear(input_size, int(input_size/4))
        self.i22i3 = torch.nn.Linear(int(input_size/4), int(input_size/16))
        self.i32i4 = torch.nn.Linear(int(input_size/16), int(input_size/64))
        self.i42i5 = torch.nn.Linear(int(input_size/64), int(input_size/128))
        self.i52o = torch.nn.Linear(int(input_size/128), output_size)
        self.act = torch.nn.ReLU()

    def forward(self, input):
        i1_z = self.i12i2(input)
        i2 = self.act(i1_z)
        i2_z = self.i22i3(i2)
        i3 = self.act(i2_z)
        i3_z = self.i32i4(i3)
        i4 = self.act(i3_z)
        i4_z = self.i42i5(i4)
        i5 = self.act(i4_z)
        i5_z = self.i52o(i5)
        o = self.act(i5_z)
        return o
    
class EFR_TX(nn.Module):
    def __init__(self, weight_matrix, utt2idx, nclass, ninp, count_speakers, nsp, nhead, nhid, nlayers, device, dropout=0.5):
        super(EFR_TX, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder, num_embeddings, embedding_dim = create_emb_layer(weight_matrix, utt2idx)
        self.ninp = ninp
        self.decoder = nn.Linear(2*ninp, nclass)
        self.speakers_embedding = lambda ix, num_class=count_speakers: torch.nn.functional.one_hot(ix, num_classes=num_class)

        self.init_weights()
        self.device = device

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for p in self.decoder.named_parameters():
            # If name of parameter ends with weight
            if p[0][-6:]=="weight":
                p[1].data.uniform_(-initrange, initrange)
            # Else it is bias
            else:
                p[1].data.zero_()

    def forward(self, src, d_ids, sp_ids, ut_len):
        device = 'cuda'
        torch.set_default_device('cuda')
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # Old Code
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # New
        src = self.encoder(src)
        new_src = torch.zeros(src.shape[0],src.shape[1],self.ninp)
        for ix1,mat in enumerate(src):
            for ix2,vec in enumerate(mat):
                new_src[ix1][ix2] = torch.cat([self.speakers_embedding(torch.tensor(sp_ids[ix1][ix2], device=device, dtype=torch.long)), src[ix1][ix2]],-1)
        src = new_src
        src = src * math.sqrt(self.ninp)        
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        decoder_ip = torch.zeros(output.size()[0],output.size()[1],output.size()[2]*2).to(self.device)
        for b in range(output.size()[0]):
            d_id = d_ids[b][0]
            main_utt = output[b][ut_len[d_id]-1]
            for s in range(ut_len[d_id]):
                this_utt = output[b][s]
                decoder_ip[b][s] = torch.cat([this_utt,main_utt],-1)
        
        output = self.decoder(decoder_ip)
        
        return decoder_ip,output