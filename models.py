import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

# Embedding the input sequence
# 把词典与dim声明，得到字典表的向量；
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # 其 该类的forward 为取值nn.Emebedding的索引；
    def forward(self, x):
        return self.embedding(x)

# The positional encoding vector
# PositionalEncoder(embedding_dim, max_seq_len, dropout)
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        # 生成位置的初始值0 ；
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                # 交替实现每个位置的值； 使用位置值的sin和cos 函数；
                pe[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        # 生成添加维度0的pe ；
        pe = pe.unsqueeze(0)
        # 将pe张量注册为模型的缓冲区buffer；
        # 模型缓冲区不会作为模型的可学习参数进行优化；但可在模型中使用
        # 通常作用于保存于模型相关的固定数据；如预先计算的常量 这里是位置信息，统计信息等。
        # self.register_buffer(name, tensor) 是 nn.Module 类的一个方法，用于注册一个张量作为模型的缓冲区。它接受两个参数 name tensor：
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x*math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        # 这里提醒我：构建的模型参数要记得to device
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        # 而且这里pe 为常量，被冻结梯度，且在缓冲区里。
        x = x + pe
        x = self.dropout(x)
        return x

# Self-attention layer
class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention
        自定义注意力；
    '''

    # 初始化只包括一个dropout层；其他没了；
    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    # 如何使用 forward的计算； 其forward中传入的q k v 都是同一个值；
    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        # 计算q / dim(k) * K 的注意力值；
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            # mask_fill 将mask中为0的元素，对应attn的张量元素替换为 -1e9.  attn张量中mask为0的位置的元素替换为-1e9
            # 为了正确执行遮蔽操作，mask的形状必须与attn的形状相匹配，根据mask进行填充；
            attn = attn.masked_fill(mask == 0, -1e9)
        # 在对attn 归一化； 执行droput 后与value 进行乘机；
        attn = self.dropout(torch.softmax(attn, dim=-1))
        # 在记性正常的矩阵乘机；
        output = torch.matmul(attn, value)

        return output
        
# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    # self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
    # 定义自注意层；
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head 多个头的维度除以头数，每个头的嵌入大小；
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections ；初始化这些投影变换；
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        # dropout 和 输出
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        # 先将 q k v 通过一个W 进行映射，这里的W不共享；
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input；将原始的q k v 分成多个头，每个头 输入多头注意力；
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        # 这一步是将转置后的张量进行内存连续化操作，即保证张量的内存布局是连续的。然后再进行视图变换
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output

# Norm layer  定义归一化层；  init 与 forward ；
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        # 对输入进行归一化的模块；类似于Batch Normalization
        # 但是他是在特征维度上进行归一化，而不是在批次维度上；
        # 这里：对样本在特征维度上的值进行归一化。参数embedding_dim指定了输入张量的特征维度
        # 即该归一化应用于输入张量的特征维度上，返回归一化后的张量。
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


# Transformer encoder layer
class EncoderLayer(nn.Module):
    # self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
    # 定义每一个encoder层；
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # 定义自注意层
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        # 定义前馈神经网络 线性 激活 线性 作为前馈神经网络；
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        # 设置dropout 与 norm
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        # Add and Muti-head attention
        # def forward(self, query, key, value, mask=None):  x2 作为query key value
        # 同时基于初始的x ； 归一化在 传入前馈网络； dropout后加上初始的x；
        x = x + self.dropout1(self.self_attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x

# Transformer decoder layer
class DecoderLayer(nn.Module):
    #初始化解码器的一层；
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 解码器的自注意与 编码器的自注意力
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        # 放在 sequential中的基本的网络层；
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        # 三个dp 和 三个 norm
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)

    # 先norm 在输入注意力 + dp -> 加上初始的编码器的部分；
    # 在norm 在编码器的注意力 + 记忆信息 + 记忆信息 + q（解码器的输出）
    # 第三次：norm + 前馈网络；这里只是解码器的一层；即每一层都整合了编码器的信息 q ；
    def forward(self, x, memory, source_mask, target_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.encoder_attention(x2, memory, memory, source_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))
        return x

# Encoder transformer
class Encoder(nn.Module):
    '''
    self.encoder = Encoder(source_vocab_size, embedding_dim, source_max_seq_len, num_heads, num_layers, dropout)
    '''
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        # 创建线性层； embedding；
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 赋值超参数
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        # nn.ModuleList创建了一个由多个EncoderLayer组成的列表。
        # nn.Modulelist 是torch的容器类，用于存储和管理神经网络模块；
        # 创建一个长度为num_layers的nn.ModuleList对象，每个对象都是encoderlayer的实例，实例共享相同的权重与参数；
        # 每个EncoderLayer实例通过传递相应的参数(embedding_dim, num_heads, 2048, dropout)来创建
        # 每个实例共享相同的权重与参数；
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
    
    def forward(self, source, source_mask):
        # Embed the source
        x = self.embedding(source)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, source_mask)
        # Normalize
        x = self.norm(x)
        return x

# Decoder transformer
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len,num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        # nn.ModuleList创建了一个由多个EncoderLayer组成的列表。
        # nn.Modulelist 是torch的容器类，用于存储和管理神经网络模块；
        # 创建一个长度为num_layers的nn.ModuleList对象，每个对象都是encoderlayer的实例，实例共享相同的权重与参数；
        # 有意义的是每个层的解码器结构都是一样的；
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
    
    def forward(self, target, memory, source_mask, target_mask):
        # Embed the source
        x = self.embedding(target)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers 在每个层进行传播；
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        # Normalize
        x = self.norm(x)
        return x


# Transformers
# 基于nn.module 定义transformer模型；
class Transformer(nn.Module):
    '''
        embedding_dim=configs["embedding_dim"],
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        num_layers=configs["n_layers"],
        num_heads=configs["n_heads"],
        dropout=configs["dropout"]
    '''
    def __init__(self, source_vocab_size, target_vocab_size, source_max_seq_len, target_max_seq_len, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # 赋值完上面的超参数；然后基于超参数设置编码器与解码器；
        self.encoder = Encoder(source_vocab_size, embedding_dim, source_max_seq_len, num_heads, num_layers, dropout)
        self.decoder = Decoder(target_vocab_size, embedding_dim, target_max_seq_len, num_heads, num_layers, dropout)
        # 在最后面添加一个线性层与dropout
        self.final_linear = nn.Linear(embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, source, target, source_mask, target_mask):
        # Encoder forward pass
        memory = self.encoder(source, source_mask)
        # Decoder forward pass
        output = self.decoder(target, memory, source_mask, target_mask)
        # Final linear layer
        # 添加一个线性层与dropout; 然后输出；
        output = self.dropout(output)
        output = self.final_linear(output)
        return output

    # 和pad id 不相等的地方是 真实数据 对应保存一个mask 矩阵；
    def make_source_mask(self, source_ids, source_pad_id):
        return (source_ids != source_pad_id).unsqueeze(-2)

    #
    def make_target_mask(self, target_ids):
        batch_size, len_target = target_ids.size()
        # 对target 数据；
        # 用于遮盖未来标记的掩码：subsequent_mask的作用是在解码过程中，防止模型在生成每个目标标记时能够看到后续的标记信息
        # 这样模型就可以知道当前的序列来预测接下来的序列；掩码来确保模型只能利用已生成的部分信息进行预测，避免利用未来信息；
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask