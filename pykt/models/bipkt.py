import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.distributions import Categorical
from .iekt_utils import mygru,funcs
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

class bipktnet(nn.Module):
    def __init__(self, skill_max, pro_max, d, dropout, n_blocks, d_ff=256,
                 l2=1e-5, num_layers=2, nheads=4, seq_len=200,
                 loss1=0.5, loss2=0.5, loss3=0.5, final_fc_dim=512,
                 separate_qa=False, start=50, kq_same=1, final_fc_dim2=256,
                 num_attn_heads=8, emb_type="qc_merge"):
        super().__init__()
        self.device = device
        self.pro_max = pro_max
        self.skill_max = skill_max
        self.model_name = "bipkt"
        self.emb_type = emb_type
        self.emb_size = d
        self.d = d
        self.seq_len = seq_len
        self.nheads = nheads

        # 论文里的超参数
        self.gamma = 0.93
        self.lambda_q = 0.1   # score loss
        self.alpha = 0.01     # question branch RL
        self.beta = 0.01      # global branch RL
        self.sigma = 0.1      # 用于标准化 score reward

        self.que_emb = QueEmb(
            num_q=pro_max,
            num_c=skill_max,
            emb_size=d,
            emb_type="qc_merge",
            model_name=self.model_name,
            device=device
        )

        self.status_embed = nn.Embedding(11, d)
        self.score_embed = nn.Embedding(11, d)
        self.diff_embed = nn.Embedding(11, d)

        self.checker_emb = funcs(10, d * 14, 10, dropout)
        self.pro_checker_emb = funcs(10, d * 14, 10, dropout)
        self.select_preemb = funcs(10, d * 4, 10, dropout)
        self.pro_select_preemb = funcs(10, d * 4, 10, dropout)

        self.cog_matrix = nn.Parameter(torch.randn(10, d * 2), requires_grad=True)
        self.acq_matrix = nn.Parameter(torch.randn(10, d * 2), requires_grad=True)
        self.pro_cog_matrix = nn.Parameter(torch.randn(10, d * 2), requires_grad=True)
        self.pro_acq_matrix = nn.Parameter(torch.randn(10, d * 2), requires_grad=True)

        self.predictor = funcs(10, d * 5, d, dropout)
        self.pro_predictor = funcs(10, d * 5, d, dropout)

        self.gru_h = mygru(0, d * 4, d)
        self.pro_gru_h = mygru(0, d * 4, d)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.time_embed = nn.Parameter(torch.rand(seq_len, d))

        # correctness head: [mG ⊕ hGv ⊕ hQv] 的简化实现
        self.out1 = nn.Sequential(
            nn.Linear(2 * d, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_fc_dim2, 1)
        )

        # score head: [mQ ⊕ hQv]
        self.out2 = nn.Sequential(
            nn.Linear(d, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_fc_dim2, 1)
        )

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.skill_max+1 and self.skill_max > 0:
                torch.nn.init.constant_(p, 0.)

    
    def get_ques_representation(self, q, c):
        """Get question representation equation 3

        Args:
            q (_type_): question ids
            c (_type_): concept ids

        Returns:
            _type_: _description_
        """
       
        v = self.que_emb(q,c)

        return v

    
    def obtain_v(self, q, c, h, emb):
        v = self.get_ques_representation(q, c)
        predict_x = torch.cat([h, v], dim=1)
        h_v = torch.cat([h, v], dim=1)
        prob = self.predictor(torch.cat([predict_x, emb], dim=1))
        return h_v, v, prob

    def pro_obtain_v(self, q, c, h, emb):
        v = self.get_ques_representation(q, c)
        predict_x = torch.cat([h, v], dim=1)
        h_v = torch.cat([h, v], dim=1)
        prob = self.pro_predictor(torch.cat([predict_x, emb], dim=1))
        return h_v, v, prob
    
    def update_state(self, h, v, emb, operate):
        """_summary_

        Args:
            h (_type_): rnn鐨刪
            v (_type_): question 琛ㄧず
            emb (_type_): s_t knowledge acquistion sensitivity
            operate (_type_): label

        Returns:
            next_p_state {}: _description_
        """
        #equation 13
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.emb_size * 2)),
            v.mul((1 - operate).repeat(1, self.emb_size * 2))], dim = 1)#v_t鎵╁睍锛屽垎鍒搴旀纭殑閿欒鐨勬儏锟??
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.emb_size * 2)),
            emb.mul((operate).repeat(1, self.emb_size * 2))], dim = 1)# s_t 鎵╁睍锛屽垎鍒搴旀纭殑閿欒鐨勬儏锟??
        inputs = v_cat + e_cat#璧峰埌concat浣滅敤
        
        h_t_next = self.gru_h(inputs, h)
        return h_t_next
    
    def pro_update_state(self, h, v, emb, operate):
        """_summary_

        Args:
            h (_type_): rnn鐨刪
            v (_type_): question 琛ㄧず
            emb (_type_): s_t knowledge acquistion sensitivity
            operate (_type_): label

        Returns:
            next_p_state {}: _description_
        """
        #equation 13
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.emb_size * 2)),
            v.mul((1 - operate).repeat(1, self.emb_size * 2))], dim = 1)#v_t鎵╁睍锛屽垎鍒搴旀纭殑閿欒鐨勬儏锟??
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.emb_size * 2)),
            emb.mul((operate).repeat(1, self.emb_size * 2))], dim = 1)# s_t 鎵╁睍锛屽垎鍒搴旀纭殑閿欒鐨勬儏锟??
        inputs = v_cat + e_cat#璧峰埌concat浣滅敤
        
        h_t_next = self.pro_gru_h(inputs, h)
        return h_t_next
    
    def pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.select_preemb(x), dim = softmax_dim)
    
    def pro_pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.pro_select_preemb(x), dim = softmax_dim)
    
    def pi_sens_func(self, x, softmax_dim = 1):
        return F.softmax(self.checker_emb(x), dim = softmax_dim)
    
    def pro_pi_sens_func(self, x, softmax_dim = 1):
        return F.softmax(self.pro_checker_emb(x), dim = softmax_dim)
    
    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def out(self, pro_logits,logits):
        out1 = self.out1(torch.cat([pro_logits, logits],dim = -1))
        out2 = self.out2(pro_logits)
        return out1, out2







class bipkt(QueBaseModel):
    def __init__(self, skill_max, pro_max, d, dropout, n_blocks, d_ff=256, 
                 l2=1e-5, num_layers=2, nheads=4, seq_len=200, loss1=0.5, loss2=0.5, loss3=0.5, final_fc_dim=512, 
                 separate_qa=False, start=50, kq_same=1, final_fc_dim2=256, num_attn_heads=8, emb_type="qc_merge"):
        model_name = "bipkt"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path="",pretrain_dim=768,device="device",seed=0)
        self.model = bipktnet(skill_max, pro_max, d, dropout, n_blocks, d_ff=256, 
                    l2=1e-5, num_layers=2, nheads=4, seq_len=200, loss1=0.5, loss2=0.5, loss3=0.5, final_fc_dim=512, 
                    separate_qa=False, start=50, kq_same=1, final_fc_dim2=256, num_attn_heads=8, emb_type="qc_merge")
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.eval_result = {}
        self.device = device
    def batch_to_device(self,data,process=True):
        dcur = data
        # q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
        # qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        # m, sm = dcur["masks"], dcur["smasks"]
        data_new = {}
        data_new['cq'] = torch.cat((dcur["qseqs"][:,0:1], dcur["shft_qseqs"]), dim=1).to(self.device)
        data_new['cc'] = torch.cat((dcur["cseqs"][:,0:1],  dcur["shft_cseqs"]), dim=1).to(self.device)
        data_new['cr'] = torch.cat((dcur["rseqs"][:,0:1], dcur["shft_rseqs"]), dim=1).to(self.device)
        data_new['ct'] = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1).to(self.device)
        data_new['cscore'] = torch.cat((dcur["score"][:,0:1], dcur["shft_score"]), dim=1).to(self.device)
        data_new['csmc'] = torch.cat((dcur["smc"][:,0:1],  dcur["shft_smc"]), dim=1).to(self.device)
        data_new['cstc'] = torch.cat((dcur["stc"][:,0:1], dcur["shft_stc"]), dim=1).to(self.device)
        data_new['cstatus'] = torch.cat((dcur["status"][:,0:1], dcur["shft_status"]), dim=1).to(self.device)
        data_new['cproblem_difficulty'] = torch.cat((dcur["problem_difficulty"][:,0:1], dcur["shft_problem_difficulty"]), dim=1).to(self.device)
        
        data_new['q'] = dcur["qseqs"].to(self.device)
        data_new['c'] = dcur["cseqs"].to(self.device)
        data_new['r'] = dcur["rseqs"].to(self.device)
        data_new['t'] = dcur["tseqs"].to(self.device)
        data_new['score'] = dcur["score"].to(self.device)
        data_new['smc'] = dcur["smc"].to(self.device)
        data_new['stc'] = dcur["stc"].to(self.device)
        data_new['status'] = dcur["status"].to(self.device)
        data_new['problem_difficulty'] = dcur["problem_difficulty"].to(self.device)
        
        
        
        
        
        data_new['qshft'] = dcur["shft_qseqs"].to(self.device)
        data_new['cshft'] = dcur["shft_cseqs"].to(self.device)
        data_new['rshft'] = dcur["shft_rseqs"].to(self.device)
        data_new['tshft'] = dcur["shft_tseqs"].to(self.device)
        data_new['scoreshft'] = dcur["shft_score"].to(self.device)
        data_new['smcshft'] = dcur["shft_smc"].to(self.device)
        data_new['stcshft'] = dcur["shft_stc"].to(self.device)
        data_new['statusshft'] = dcur["shft_status"].to(self.device)
        data_new['problem_difficultyshft'] = dcur["shft_problem_difficulty"].to(self.device)
        
        
        
        data_new['m'] = dcur["masks"].to(self.device)
        data_new['sm'] = dcur["smasks"].to(self.device)

        return data_new

    def train_one_step(self,data,process=True,return_all=False):
           # self.step+=1
        # debug_print(f"step is {self.step},data is {data}","train_one_step")
        # debug_print(f"step is {self.step}","train_one_step")
        BCELoss = torch.nn.BCEWithLogitsLoss()
        
        data_new, emb_action_list, p_action_list, states_list, pre_state_list,reward_list, predict_list, ground_truth_list,pro_emb_action_list, pro_p_action_list, pro_states_list, pro_pre_state_list,score_reward_list, pro_predict_list, ground_score_list = self.predict_one_step(data,process=process)
        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]

        #浠ヤ笅鏄己鍖栧涔犻儴鍒嗗唴锟??
        seq_num = torch.where(data['qseqs']!=0,1,0).sum(axis=-1)+1
        emb_action_tensor = torch.stack(emb_action_list, dim = 1)
        p_action_tensor = torch.stack(p_action_list, dim = 1)
        state_tensor = torch.stack(states_list, dim = 1)
        pre_state_tensor = torch.stack(pre_state_list, dim = 1)
        reward_tensor = torch.stack(reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, seq_len)).float().to(self.device)#equation15
        logits_tensor = torch.stack(predict_list, dim = 1)
        # score_reward_tensor = torch.stack(score_reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, seq_len)).float()
        ground_truth_tensor = torch.stack(ground_truth_list, dim = 1)
        
        pro_emb_action_tensor = torch.stack(pro_emb_action_list, dim = 1)
        pro_p_action_tensor = torch.stack(pro_p_action_list, dim = 1)
        pro_state_tensor = torch.stack(pro_states_list, dim = 1)
        pro_pre_state_tensor = torch.stack(pro_pre_state_list, dim = 1)
        score_reward_tensor = torch.stack(score_reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, seq_len)).float().to(self.device)
        pro_logits_tensor = torch.stack(pro_predict_list, dim = 1)
        ground_score_tensor = torch.stack(ground_score_list, dim = 1)
        
        
        
        loss = []
        loss2 = []
        tracat_logits = []
        tracat_ground_truth = []
        
        pro_tracat_logits = []
        pro_tracat_ground_truth = []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
            this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)
            this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)

            td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_cog = td_target_cog
            delta_cog = delta_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_sens = td_target_sens
            delta_sens = delta_sens.detach().cpu().numpy()

            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]#equation17
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)


            pi_cog = self.model.pi_cog_func(this_cog_state[:-1])
            pi_a_cog = pi_cog.gather(1,p_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_cog = -torch.log(pi_a_cog) * advantage_cog#equation16
            
            loss.append(torch.sum(loss_cog))

            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                # advantage = args.gamma * args.beta * advantage + delta_t[0]
                advantage = self.model.gamma * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)

            pi_sens = self.model.pi_sens_func(this_sens_state[:-1])
            
            
            pi_a_sens = pi_sens.gather(1,emb_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_sens = - torch.log(pi_a_sens) * advantage_sens#equation18
            loss.append(torch.sum(loss_sens))
            

            this_prob = logits_tensor[i][0: this_seq_len]
            this_groud_truth = ground_truth_tensor[i][0: this_seq_len]

            tracat_logits.append(this_prob)
            tracat_ground_truth.append(this_groud_truth)
            
            #闂锟??
            this_cog_state = torch.cat([pro_pre_state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, pro_pre_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)
            this_sens_state = torch.cat([pro_state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, pro_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)
            this_reward_list = score_reward_tensor[i]
            
            td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_cog = td_target_cog
            delta_cog = delta_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_sens = td_target_sens
            delta_sens = delta_sens.detach().cpu().numpy()

            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]#equation17
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)

            pi_cog = self.model.pro_pi_cog_func(this_cog_state[:-1])
            pi_a_cog = pi_cog.gather(1,pro_p_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_cog = -torch.log(pi_a_cog) * advantage_cog#equation16
            
            loss2.append(torch.sum(loss_cog))

            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                # advantage = args.gamma * args.beta * advantage + delta_t[0]
                advantage = self.model.gamma * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)

            pi_sens = self.model.pro_pi_sens_func(this_sens_state[:-1])
            pi_a_sens = pi_sens.gather(1,pro_emb_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_sens = - torch.log(pi_a_sens) * advantage_sens#equation18
            loss2.append(torch.sum(loss_sens))
            

            this_prob = pro_logits_tensor[i][0: this_seq_len]
            this_groud_truth = ground_score_tensor[i][0: this_seq_len]

            pro_tracat_logits.append(this_prob)
            pro_tracat_ground_truth.append(this_groud_truth)

            
            bce = BCELoss(
                torch.cat(tracat_logits, dim=0),
                torch.cat(tracat_ground_truth, dim=0).float()
            )

            loss_fn = torch.nn.MSELoss()
            pred = torch.cat(pro_tracat_logits, dim=0).view(-1)
            truth = torch.cat(pro_tracat_ground_truth, dim=0).view(-1).float()
            mse = loss_fn(pred, truth)

            y = torch.cat(tracat_logits, dim=0)
            label_len = torch.cat(tracat_ground_truth, dim=0).size(0)

            global_rl = sum(loss) / label_len
            question_rl = sum(loss2) / label_len

            total_loss = (
                bce
                + self.model.lambda_q * mse
                + self.model.beta * global_rl
                + self.model.alpha * question_rl
            )

            return y, total_loss
        
    


        
    def predict_one_step(self, data, return_details=True, process=True, return_raw=False):
        sigmoid_func = torch.nn.Sigmoid()
        data_new = self.batch_to_device(data, process)

        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]
        batch_index = torch.arange(data_len, device=self.device)

        # global state
        h = torch.zeros(data_len, self.model.emb_size, device=self.device)

        # question-level state memory
        # 第0列保留给“从未出现过”
        pro_h_list = torch.zeros(data_len, seq_len + 1, self.model.d, device=self.device)
        last_pro_time = torch.zeros((data_len, self.model.pro_max), dtype=torch.long, device=self.device)

        uni_prob_list, states_list, reward_list, score_reward_list = [], [], [], []
        p_action_list, pre_state_list, emb_action_list = [], [], []
        predict_list, ground_truth_list = [], []

        pro_p_action_list, pro_pre_state_list, pro_emb_action_list = [], [], []
        pro_states_list, pro_predict_list, ground_score_list = [], [], []

        for seqi in range(seq_len):
            now_item_pro = data_new['cq'][:, seqi]
            pro_diff_emb = self.model.diff_embed(data_new['cproblem_difficulty'][:, seqi])

            prev_idx = last_pro_time[batch_index, now_item_pro]
            pro_h = pro_h_list[batch_index, prev_idx]

            # CE 输入
            ques_h = torch.cat([
                self.model.get_ques_representation(q=data_new['cq'][:, seqi], c=data_new['cc'][:, seqi]),
                h,
                pro_diff_emb
            ], dim=1)

            pro_ques_h = torch.cat([
                self.model.get_ques_representation(q=data_new['cq'][:, seqi], c=data_new['cc'][:, seqi]),
                pro_h,
                pro_diff_emb
            ], dim=1)

            # global CE
            flip_prob_emb = self.model.pi_cog_func(ques_h)
            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = self.model.cog_matrix[emb_ap, :]

            # question CE
            pro_flip_prob_emb = self.model.pro_pi_cog_func(pro_ques_h)
            pro_m = Categorical(pro_flip_prob_emb)
            pro_emb_ap = pro_m.sample()
            pro_emb_p = self.model.pro_cog_matrix[pro_emb_ap, :]

            # 预测特征
            h_v, v, logits_feat = self.model.obtain_v(
                q=data_new['cq'][:, seqi],
                c=data_new['cc'][:, seqi],
                h=h,
                emb=emb_p
            )

            pro_h_v, pro_v, pro_logits_feat = self.model.pro_obtain_v(
                q=data_new['cq'][:, seqi],
                c=data_new['cc'][:, seqi],
                h=pro_h,
                emb=pro_emb_p
            )

            # score / status
            ground_score = data_new['cscore'][:, seqi].float()
            score_norm = ground_score / 10.0
            ground_score_emb = self.model.score_embed(data_new['cscore'][:, seqi].long())
            ground_status = self.model.status_embed(data_new['cstatus'][:, seqi].long())

            # out1: correctness logit, out2: score prediction
            correctness_logit, score_pred = self.model.out(pro_logits_feat, logits_feat)
            prob = sigmoid_func(correctness_logit)
            score_pred = sigmoid_func(score_pred).squeeze(1)


            ground_truth = data_new['cr'][:, seqi]
            out_operate_groundtruth = ground_truth.unsqueeze(-1).float()

            # FR: ground-truth response rep
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size(-1))),
                h_v.mul((1 - out_operate_groundtruth).repeat(1, h_v.size(-1)))
            ], dim=1)

            pro_out_x_groundtruth = torch.cat([
                pro_h_v.mul(out_operate_groundtruth.repeat(1, pro_h_v.size(-1))),
                pro_h_v.mul((1 - out_operate_groundtruth).repeat(1, pro_h_v.size(-1)))
            ], dim=1)

            # FR: predicted response rep
            out_operate_logits = (prob > 0.5).float()

            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size(-1))),
                h_v.mul((1 - out_operate_logits).repeat(1, h_v.size(-1)))
            ], dim=1)

            pro_out_x_logits = torch.cat([
                pro_h_v.mul(out_operate_logits.repeat(1, pro_h_v.size(-1))),
                pro_h_v.mul((1 - out_operate_logits).repeat(1, pro_h_v.size(-1)))
            ], dim=1)

            out_x = torch.cat([out_x_groundtruth, out_x_logits, ground_status, ground_score_emb], dim=1)
            pro_out_x = torch.cat([pro_out_x_groundtruth, pro_out_x_logits, ground_status, ground_score_emb], dim=1)

            # global FR
            flip_prob_emb = self.model.pi_sens_func(out_x)
            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.model.acq_matrix[emb_a, :]

            # question FR
            pro_flip_prob_emb = self.model.pro_pi_sens_func(pro_out_x)
            pro_m = Categorical(pro_flip_prob_emb)
            pro_emb_a = pro_m.sample()
            pro_emb = self.model.pro_acq_matrix[pro_emb_a, :]

            # 状态更新
            h = self.model.update_state(h, v, emb, ground_truth.unsqueeze(1).float())
            pro_h = self.model.pro_update_state(pro_h, pro_v, pro_emb, ground_truth.unsqueeze(1).float())

            # 注意：这里存 seqi + 1，避免和“未出现过=0”冲突
            last_pro_time[batch_index, now_item_pro] = seqi + 1
            pro_h_list[:, seqi + 1] = pro_h

            # reward
            this_reward = (out_operate_logits.squeeze(1) == ground_truth.float()).float()
            score_reward = torch.exp(-((score_pred - score_norm) ** 2) / (2 * (self.model.sigma ** 2)))

            # 收集
            uni_prob_list.append(prob.detach())
            emb_action_list.append(emb_a)
            p_action_list.append(emb_ap)
            states_list.append(out_x)
            pre_state_list.append(ques_h)
            predict_list.append(prob.squeeze(1))
            ground_truth_list.append(ground_truth)

            pro_emb_action_list.append(pro_emb_a)
            pro_p_action_list.append(pro_emb_ap)
            pro_states_list.append(pro_out_x)
            pro_pre_state_list.append(pro_ques_h)
            pro_predict_list.append(score_pred)
            ground_score_list.append(score_norm)

            reward_list.append(this_reward)
            score_reward_list.append(score_reward)

        prob_tensor = torch.cat(uni_prob_list, dim=1)

        if return_details:
            return (
                data_new, emb_action_list, p_action_list, states_list, pre_state_list,
                reward_list, predict_list, ground_truth_list,
                pro_emb_action_list, pro_p_action_list, pro_states_list, pro_pre_state_list,
                score_reward_list, pro_predict_list, ground_score_list
            )
        else:
            return prob_tensor[:, 1:]
