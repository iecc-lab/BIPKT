import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
from .que_base_model import QueBaseModel,QueEmb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class ZYWnet(nn.Module):
    def __init__(self, skill_max, pro_max, d, dropout, n_blocks, d_ff=256, 
                 l2=1e-5, num_layers=2, nheads=4, seq_len=200, loss1=0.5, loss2=0.5, loss3=0.5, final_fc_dim=512, 
                 separate_qa=False, start=50, kq_same=1, final_fc_dim2=256, num_attn_heads=8, emb_type="qc_merge"):
        super().__init__()

        self.pro_max = pro_max
        self.skill_max = skill_max
        self.model_name = "zyw_que"
        self.emb_type = emb_type
        self.dropout = dropout
        self.que_emb = QueEmb(num_q=pro_max,num_c=skill_max,emb_size=128,emb_type="zyw",model_name=self.model_name,device=device)

        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))
        self.score_embed = nn.Parameter(torch.rand(11, d))
        self.ans_embed = nn.Parameter(torch.rand(2, d))
        self.status_embed = nn.Parameter(torch.rand(11, d))
        self.pd_embed = nn.Parameter(torch.rand(11, d))
        self.problem_difficulty_emb = nn.Parameter(torch.rand(11, d*2))
        self.out = nn.Sequential(
            nn.Linear(384, d),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d, 1)
        )
        self.kq_same = 1
        self.skill_max = skill_max
        self.l2 = 1e-5
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type



        self.dropout = nn.Dropout(p=dropout)

        self.time_embed = nn.Parameter(torch.rand(200, d))#序列长度�?200 时间间隔最�?200

        self.ls_state = nn.Parameter(torch.rand(1, d))#？？？？领域状�?
        self.c_state = nn.Parameter(torch.rand(1, d))#？？？？未使用到

        self.pro_state = nn.Parameter(torch.rand(199, d))#序列200，只用算�?199的知识状�?
        self.skill_state = nn.Parameter(torch.rand(199, d))#序列200，只用算�?199的知识状�?
        
        

        self.obtain_pro_forget = nn.Sequential(#问题状态更新模�?
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.obtain_pro_state = nn.Sequential(#问题状态更新模�?
            nn.Linear(3 *d , d)
        )
        


        if self.skill_max > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.skill_max+1, 1) # 题目难度
            else:
                self.difficult_param = nn.Embedding(self.skill_max+1, d) # 题目难度
            self.q_embed_diff = nn.Embedding(self.pro_max+1, d) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.pro_max + 1, d *2) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # pro_max+1 ,d_model
            self.q_embed = nn.Embedding(self.pro_max, d)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.pro_max+1, d)
            else: # false default
                self.qa_embed = nn.Embedding(2, d*2)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(pro_max=pro_max, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=2 * d, d_feature=d / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out2 = nn.Sequential(
            nn.Linear(3 * d,
                      final_fc_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(0.3),
            nn.Linear(final_fc_dim2, 1)
        )
        self.out1 = nn.Sequential(
            nn.Linear(896,
                      final_fc_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(0.3),
            nn.Linear(final_fc_dim2, 1)
        )

        self.reset()
    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.skill_max+1 and self.skill_max > 0:
                torch.nn.init.constant_(p, 0.)



    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)





        
    def forward(self, dcur, qtest=False, train=True):
        """
        last_*:表示去掉序列最后面元素
        next_*:表示去掉序列最前面元素
        """
        last_problem, last_skill, last_ans = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        next_problem, next_skill, next_ans = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        last_score, last_smc, last_stc, last_status, last_problem_difficult = dcur["score"].long(), dcur["smc"].long(), dcur["stc"].long(), dcur["status"].long(), dcur["problem_difficulty"].long()
        next_score, next_smc, next_stc, next_status, next_problem_difficult = dcur["shft_score"].long(), dcur["shft_smc"].long(), dcur["shft_stc"].long(), dcur["shft_status"].long(), dcur["shft_problem_difficulty"].long()

        
        
        device = last_problem.device
        batch = last_problem.shape[0]
        seq = last_problem.shape[-1]

        xemb,emb_qca,emb_qc,emb_q,emb_c  = self.que_emb(next_problem, next_skill)

        next_pro_embed = torch.concat([emb_qc,F.embedding(next_problem_difficult.long(), self.pd_embed)], dim=-1)

        next_X = next_pro_embed + torch.concat([F.embedding(next_ans.long(), self.ans_embed), F.embedding(next_score.long(), self.score_embed)], dim=-1)#Xt+1 = Et+1 + Rrt+1

        last_pro_time = torch.zeros((batch, self.pro_max)).to(device)  # batch pro

        pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d 问题状�?


        last_pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d


        batch_index = torch.arange(batch).to(device)


        all_time_gap = torch.ones((batch, seq)).to(device)
        all_time_gap_embed = F.embedding(all_time_gap.long(), self.time_embed)  # batch seq d

        res_p = []
        concat_q = []
        pro_state_list = []
        for now_step in range(seq):

            now_pro_embed = next_pro_embed[:, now_step]  # batch d

            now_item_pro = next_problem[:, now_step]  # batch
            
            last_batch_pro_time = last_pro_time[batch_index, now_item_pro]  # batch
            last_batch_pro_state = pro_state[batch_index, last_batch_pro_time.long()]  # batch d

            
            time_gap = now_step - last_batch_pro_time  # batch
            time_gap_embed = F.embedding(time_gap.long(), self.time_embed)  # batch d 问题时间间隔嵌入


            item_pro_state_forget = self.obtain_pro_forget(
                self.dropout(torch.cat([last_batch_pro_state, time_gap_embed], dim=-1))) #遗忘模块 ft = Sigmoid([Zt−�? �? Iα]W1 + b1)
            last_batch_pro_state = last_batch_pro_state * item_pro_state_forget
            
            last_pro_state[:, now_step] = last_batch_pro_state
    
            
 
            final_state = torch.cat(
                [ last_batch_pro_state, now_pro_embed], dim=-1)
    
            pro_state_list.append(final_state)
   
            

        

            pro_get = next_X[:, now_step]
            
            item_pro_obtain = self.obtain_pro_state(
                self.dropout(torch.cat([last_batch_pro_state, pro_get], dim=-1)))
            item_pro_state = last_batch_pro_state + torch.tanh(item_pro_obtain)
            
            
            last_pro_time[batch_index, now_item_pro] = now_step
            pro_state[:, now_step] = item_pro_state

        final_pro_state = torch.stack(pro_state_list, dim=1)
       
        
        qd_embed = F.embedding(next_problem_difficult.long(), self.problem_difficulty_emb)
        qs_emb = F.embedding(next_score.long(), self.score_embed)
        qstatus = F.embedding(next_status.long(), self.status_embed)



        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        q_embed_data = next_pro_embed
        qa_embed_data = next_X
        y2, y3 = 0, 0
        emb_type = self.emb_type
    
        d_output = self.model(q_embed_data, qa_embed_data)
        concat_q = torch.cat([d_output, final_pro_state, q_embed_data], dim=-1)
    
        output = self.out1(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        final_pro_state = self.out2(final_pro_state).squeeze(-1)
        
        if train:
            return preds, final_pro_state
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, pro_max,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])


        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目�?
            # print(x[0,0,:])
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value�?0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score�?0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)





class ZYW_QUE(QueBaseModel):
    def __init__(self, skill_max, pro_max, d, dropout, n_blocks, d_ff=256, 
                 l2=1e-5, num_layers=2, nheads=4, seq_len=200, loss1=0.5, loss2=0.5, loss3=0.5, final_fc_dim=512, 
                 separate_qa=False, start=50, kq_same=1, final_fc_dim2=256, num_attn_heads=8, emb_type="qc_merge"):
        model_name = "zyw_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path="",pretrain_dim=768,device="cpu",seed=0)
        self.model = ZYWnet(skill_max, pro_max, d, dropout, n_blocks, d_ff=256, 
                    l2=1e-5, num_layers=2, nheads=4, seq_len=200, loss1=0.5, loss2=0.5, loss3=0.5, final_fc_dim=512, 
                    separate_qa=False, start=50, kq_same=1, final_fc_dim2=256, num_attn_heads=8, emb_type="qid")
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.eval_result = {}
    


    def train_one_step(self,data,process=True,return_all=False):
        outputs,final_pro_state,data = self.predict_one_step(data,return_details=True,process=process)
        
        sm = data["smasks"]
        y = torch.masked_select(outputs[0], sm)
        t = torch.masked_select(data["shft_rseqs"], sm)
        socre = torch.masked_select(data["score"], sm)

        final_pro_state = torch.masked_select(final_pro_state[0], sm)
        loss1 = binary_cross_entropy(y.float(), t.float())
        loss_fn = torch.nn.MSELoss()
        # loss3 = loss_fn(pd.double(), (socre / 10).double())                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        loss2 = loss_fn(y.double(), (socre / 10).double())
        loss3 = loss_fn(final_pro_state.double(), (socre / 10).double())


        loss =  0.01 * loss3 + loss1
        return y,loss



    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        outputs,final = self.model(data)

        if return_details:
            return outputs,final,data
        else:
            return outputs,final,data