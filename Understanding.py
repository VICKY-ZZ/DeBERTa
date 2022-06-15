# 参考博客：https://yam.gift/2020/06/27/Paper/2020-06-27-DeBERTa/
class DientangledSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/config.num_attention_heads)
        # 为啥不直接等于hidden_size呢？？？
        self.all_head_size = self.num_attention_heads*self.attention_head_size
        # Wq,c,生成Qc，变成三份--QKV
        self.in_proj = torch.nn.Linear(config.hidden_size,self.all_head_size*3, bias = False)
        self.q_bias = torch.nn.Parameter(
            torch.zeros((self.all_head_size),dtype=torch.float)
        )
        self.v_bias = torch.nn.Parameter(
            torch.zeros((self.all_head_size), dtype = torch.float)
        )

        # ----------pos篇
        self.pos_att_type = ['p2c','c2p']
        self.max_relative_positions = config.max_relative_positions
        # pos的dropout for what???
        self.pos_dropout = StableDropout(config.hidden_dropout_prob)
        self.pos_proj = torch.nn.Linear(config.hidden_size,self.all_head_size)
        self.dropout = StableDropout(config.attention_probs_dropout_prob)
    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1]+(self.num_attention_heads,-1)
        # 相当于不要x的最后一维（应该是hidden_size），然后换成num_att_head,每个head大小(head_size)
        x = x.view(*new_x_shape)
        # (batch_size,num_heads,seq_len,head_size*3)
        return x.permute(0,2,1,3)
    def forward(self, hidden_states, attention_mask,
                return_att=False, query_states=None,
                relative_pos=None, rel_embeddings=None):
        # hidden_states是前一层传过来的attention(Q,K,V)
        # attention_mask的shape：[B,N,N],[i,j]:=第i个token attend第j个token
        # return_att是否返回注意力矩阵A
        # query_states是q的state,Qc=HWq,c;Qr=PWq,r
        # relative_pos.shape=[B,N,N],范围在max_relative_positions之内
        # rel_embeddings.shape=[2*max_relative_positions,hidden_size]---hidden_size是一个向量有多长，前面是一共有一个相对位置向量

        # (batch_size,seq_len,hidden_size*3)---Q,K,V
        # qp--in_proj对应大matrix,QKV一次计算
        qp = self.in_proj(hidden_states)
        # (batch_size,seq_len,num_att_heads,att_size*3)---permute-->
        # (batch_size,num_att_heads,  seq_len, att_head_size * 3)---chunk-->
        # 3*(batch_size,num_att_heads, seq_len,att_head_size)
        query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3,dim=-1)
        # 为啥key没有bias???
        # [1,1,nums_heads,head_size]--->[1,nums_heads,1,att_head_size]
        query_layer += self.transpose_for_scores(self.q_bias.unsqueeze(0).unsqueeze(0))
        value_layer += self.transpose_for_scores(self.v_bias.unsqueeze(0).unsqueeze(0))

        rel_att = None
        scale_factor = 1
        if 'c2q' in self.pos_att_type:
            scale_factor+=1
        if 'p2c' in self.pos_att_type:
            scale_factor+=1
        if 'p2p' in self.pos_att_type:
            scale_factor+=1
        # att_head_size*scale_factor  for what???为什么用head_size和scale_factor做scale？
        scale = math.sqrt(query_layer.size(-1)*scale_factor)
        query_layer = query_layer/scale
        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))

        # 比bert多的计算att score part
        rel_embeddings = self.pos_dropout(rel_embeddings)
        rel_att = self.disentangled_att_bias(
            query_layer,key_layer,relative_pos,rel_embeddings,scale_factor
        )

        attention_scores = attention_scores+rel_att

        attention_probs = XSoftmax.apply(attention_scores,attention_mask,-1)
        attention_probs = self.dropout(attention_probs)
        # (batch_size, num_att_heads, seq_len, att_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # 用view前需先用continuous
        # (batch_size, seq_len, num_att_heads, att_head_size)
        context_layer = context_layer.permute(0,2,1,3).continuous()
        # (batch_size, seq_len,num_att_heads*att_head_size)
        new_context_layer_shape = context_layer.size()[:-2]+(-1,)
        # (batch_size, seq_len,all_heads_size),变回了x的样子
        context_layer = context_layer.view(*new_context_layer_shape)

        # attention_probs是dropout过后的attention_probs
        return (context_layer,attention_probs)

    def disentangled_att_bias(self,
                              query_layer,
                              key_layer,
                              relative_pos,
                              rel_embeddings,
                              scale_factor):
        # query_layer.shape=(batch_size,num_att_heads,query_sen_len,att_head_size)
        # key_layer
        # relative_pos.shape=(1,query_size, key_size)
        # rel_embeddings.shape=(max_relative_positions*2, hidden_size)
        #为什么要scale呢
        # scale_factor:3


        relative_pos = relative_pos.unsqueeze(1)
        # 取相对位置范围 和 QK len中较小的--》QKlen超过相对位置范围便不看了
        att_span = min(max(query_layer.size(-2),key_layer.size(-2)),self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        # rel_embeddings.shape=[0:max_relative_positions*2]
        # (1,att_span*2,hidden_size)
        rel_embeddings = rel_embeddings[
            self.max_relative_positions-att_span:
                         self.max_relative_positions+att_span,:
        ].unsqueeze(0)

        if 'c2p' in self.pos_att_type:
            # 没有bias
            # (1,att_span*2,hidden_size)
            pos_key_layer = self.pos_proj(rel_embeddings)
            # (1,num_att_heads,att_span*2,att_heads_size)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
        if 'p2c' in self.pos_att_type:
            # 没有bias
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            # (1,num_att_heads, att_span*2, att_head_size)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        if 'c2p' in self.pos_att_type:
            # query_layer.shape=(batch_size,num_att_heads,query_sen_len, att_head_size)
            # pos_key_layer.shape= (1,num_att_heads,att_span*2,att_heads_size)--transpose---
            # ---->(1, num_att_heads, att_heads_size,att_span * 2)
            # return (batch_size, num_att_heads, query_sen_len,att_span * 2)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1,-2))
            # 将输入input张量每个元素的夹紧到区间[min, max][min, max]，并返回结果到一个新张量。
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
            #gather的用法？？
            c2p_att = torch.gather(c2p_att, dim=-1,index=c2p_pos.expand(
                [
                    query_layer.size(0),
                    query_layer.size(1),
                    query_layer.size(2),
                    relative_pos.size(-1)
                ]
            ))
            score += c2p_att

        if 'p2c' in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1)*scale_factor)
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1,-2))
            p2c_att = torch.gather(p2c_att, dim=-1, index = p2c_pos.expand(
                [
                    key_layer.size(0),
                    key_layer.size(1),
                    key_layer.size(2),
                    relative_pos.size(-2)
                ]
            )).transpose(-1,-2)

            score+=p2c_att
        return score

