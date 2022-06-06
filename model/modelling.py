import math

import torch
import torch.nn as nn
import  torch.nn.functional as F
from copy import deepcopy
from transformers import BertTokenizer
from torch.autograd import Variable
Config = {
        'src_vocab_size' : 30522,
        'trg_vocab_size' : 30522,
        'embed_size' : 512,
        'max_len' : 512,
        'type_vocab_size' : 2,
        'n_heads' : 8,
        'n_encoder' : 1,
        'n_decoder' :1,
        'd_ff' : 2048,
}

class Transformer(nn.Module):
    def __init__(self,Config):
        super(Transformer, self).__init__()
        self.Config = Config
        self.encoder = Encoder(Config)
        self.decoder = Decoder(Config)
        self.projection = nn.Linear(Config['embed_size'],Config['trg_vocab_size'])
    def forward(self,src,trg,src_mask,trg_mask):
        b = src.size(0)
        src = src.view(b,Config['max_len'])
        trg = trg.view(b,Config['max_len'])
        enc_outputs = self.encoder(src,src_mask)
        dec_outputs = self.decoder(enc_outputs,src_mask,trg,trg_mask)
        dec_logit = self.projection(dec_outputs)
        return dec_logit

class Embedding(nn.Module):
    def __init__(self,vocab_size,Config):
        super(Embedding, self).__init__()
        self.Config = Config
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size,Config['embed_size'])
    def forward(self,x):
        x = self.embedding(x)
        return x * math.sqrt(self.Config['embed_size'])
class PosEmbedding(nn.Module):
    def __init__(self,Config):
        super(PosEmbedding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        pe = torch.zeros(Config['max_len'],Config['embed_size'])
        position = torch.arange(0,Config['max_len']).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,Config['max_len'],2) * -(math.log(10000.0) / Config['embed_size']))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self,Config):
        super(Encoder, self).__init__()
        self.embedding = Embedding(Config['src_vocab_size'],Config)
        self.posEmbedding = PosEmbedding(Config)
        self.EncoderLayers = nn.ModuleList([EncoderLayer(Config) for _ in range(Config['n_encoder'])])
    def forward(self,src,src_mask):
        x = self.embedding(src)
        x = self.posEmbedding(x)
        for layer in self.EncoderLayers:
            x = layer(x,src_mask)
        return x
class EncoderLayer(nn.Module):
    def __init__(self,CONFIG):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(CONFIG)
        self.layerNorm1 = nn.LayerNorm(CONFIG['embed_size'])
        self.feedForward = FeedForward(CONFIG)
        self.layerNorm2 = nn.LayerNorm(CONFIG['embed_size'])
    def forward(self,x,src_mask):
        attn_outs = self.attention(x,x,x,src_mask)
        x = attn_outs + x
        x = self.layerNorm1(x)
        ff_outs = self.feedForward(x)
        x = ff_outs + x
        self.layerNorm2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,Config):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(Config['embed_size'],Config['embed_size'])
        self.WK = nn.Linear(Config['embed_size'],Config['embed_size'])
        self.WV = nn.Linear(Config['embed_size'],Config['embed_size'])
        self.scaled_dot_product_attention = Scaled_Dot_Product_Attention(Config)
        self.logit = nn.Linear(Config['embed_size'],Config['embed_size'])
        self.dropout = nn.Dropout(0.1)
        self.n_heads = Config['n_heads']
        self.d_k = Config['embed_size'] / self.n_heads
    def forward(self,Q,K,V,attn_mask=None):
        q,k,v = self.WQ(Q),self.WK(K),self.WV(V)
        attn_outs = self.scaled_dot_product_attention(q,k,v,attn_mask)
        attn_outs = attn_outs.transpose(1,2).contiguous().view(q.size(0),-1,int(self.n_heads*self.d_k))
        attn_outs = self.dropout(self.logit(attn_outs))
        return attn_outs

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self,Config):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.n_heads = Config['n_heads']
        assert Config['embed_size']  % self.n_heads == 0
        self.d_k = Config['embed_size'] // self.n_heads
        self.dropout = nn.Dropout(0.1)
    def forward(self,q,k,v,attn_mask=None):
        #(b,n,d)
        b = q.size(0)
        q = q.view(b,-1,self.n_heads,self.d_k).transpose(1,2) # b,n,k,h
        k = k.view(b,-1,self.n_heads,self.d_k).transpose(1,2)
        v = v.view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        attn_outputs = self.attention(q,k,v,attn_mask,dropout=self.dropout)

        return attn_outputs

        # b n k h
    def attention(self,q,k,v,attn_mask=None,dropout=None):
        d_k = q.size(-1)
        scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k)
        if attn_mask is not None:
            assert scores.size() == attn_mask.size()
            scores.masked_fill_(attn_mask == 0.,-1e9)
        scores = F.softmax(scores,dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores,v)

class FeedForward(nn.Module):
    def __init__(self,Config):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(Config['embed_size'],Config['d_ff'])
        self.w_2 = nn.Linear(Config['d_ff'],Config['embed_size'])
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):
        return  self.w_2(self.dropout(F.relu(self.w_1(x))))

class Decoder(nn.Module):
    def __init__(self,Config):
        super(Decoder, self).__init__()
        self.embedding = Embedding(Config['trg_vocab_size'],Config)
        self.pe = PosEmbedding(Config)
        self.DecoderLayers = nn.ModuleList([DecoderLayer(Config) for _ in range(Config['n_decoder'])])
    def forward(self,enc_outputs,src_mask,trg,trg_mask):
        x = self.embedding(trg)
        x = self.pe(x)
        for layer in self.DecoderLayers:
            x = layer(x,enc_outputs,src_mask,trg_mask)
        return x
class DecoderLayer(nn.Module):
    def __init__(self,Config):
        super(DecoderLayer, self).__init__()
        self.decMultiHeadAttention = MultiHeadAttention(Config)
        self.encDecMultiHeadAttention = MultiHeadAttention(Config)
        self.ff = FeedForward(Config)
    def forward(self,x,enc_outputs,src_mask,trg_mask):
        dec_outputs = self.decMultiHeadAttention(x,x,x,trg_mask)
        x = dec_outputs + x
        enc_dec_outputs = self.encDecMultiHeadAttention(x,enc_outputs,enc_outputs,src_mask)
        x = x + enc_dec_outputs
        ff_outs = self.ff(x)
        x = x + ff_outs
        return x

if __name__ == '__main__':
    from transformers import BertTokenizer
    en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cn_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    transformer = Transformer(Config)
    en = 'hello'
    ch = '你好呀'
    t_ch = cn_tokenizer(ch, return_tensors='pt')['input_ids']
    t_en = en_tokenizer(en,return_tensors='pt')['input_ids']
    src_mask = en_tokenizer(en, return_tensors='pt')['attention_mask']
    trg_mask = cn_tokenizer(ch, return_tensors='pt')['attention_mask']
    output = transformer(t_en,t_ch,src_mask,trg_mask)
    print(output)