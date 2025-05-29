"""
基于Transformer的中英翻译模型示例
此代码展示了如何使用Transformer架构构建一个简单的中英翻译系统
Transformer相比RNN/LSTM的主要优势是能够并行处理序列数据并更好地捕获长距离依赖

数据流程:
1. 预处理中英文句子对，分词并添加特殊标记
2. 构建词汇表并将文本转换为索引
3. 创建Transformer模型进行训练
4. 使用训练好的模型进行翻译并评估性能
"""

# 导入必要的库
import numpy as np          # 用于数值计算和数组操作
import torch                # PyTorch深度学习框架
import torch.nn as nn       # 神经网络模块
import torch.optim as optim # 优化器
import torch.nn.functional as F # 激活函数和其他功能性操作
import random               # 随机数生成，用于随机初始化和打乱数据
import time                 # 用于计时，评估训练时间
import math                 # 数学函数，如sqrt、sin、cos等
from collections import Counter # 用于词频统计
import matplotlib.pyplot as plt # 绘图库，用于可视化训练过程

# 设置随机种子，确保结果可复现
# 这对于实验比较和调试非常重要
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 简化版BLEU评分函数
def compute_bleu(reference, candidate, max_n=4):
    """
    计算BLEU分数 - 机器翻译质量评估指标
    
    参数:
        reference (str): 参考翻译（标准答案）
        candidate (str): 候选翻译（模型输出）
        max_n (int): 最大n-gram大小，默认为4
        
    返回:
        float: BLEU分数，范围0-1，越高越好
        
    工作原理:
    1. 分词，将字符串分割成单词列表
    2. 计算n-gram精度（n从1到max_n）
    3. 应用长度惩罚，防止过短翻译获得高分
    4. 计算加权几何平均得到最终分数
    """
    # 预处理：转为小写
    reference = reference.lower()
    candidate = candidate.lower()
    
    # 分词（简化起见，按空格分割）
    ref_tokens = reference.split()    # 参考翻译单词列表
    cand_tokens = candidate.split()   # 候选翻译单词列表
    
    # 如果候选翻译为空，返回0分
    if len(cand_tokens) == 0:
        return 0
    
    # 计算各阶n-gram精度
    precisions = []  # 存储1-gram到max_n-gram的精度
    for n in range(1, min(max_n, len(cand_tokens)) + 1):
        # 创建参考翻译的n-gram集合及其频率
        ref_ngrams = Counter()  # {n-gram字符串: 出现次数}
        for i in range(len(ref_tokens) - n + 1):
            ngram = ' '.join(ref_tokens[i:i+n])
            ref_ngrams[ngram] += 1
            
        # 创建候选翻译的n-gram集合及其频率
        cand_ngrams = Counter()  # {n-gram字符串: 出现次数}
        for i in range(len(cand_tokens) - n + 1):
            ngram = ' '.join(cand_tokens[i:i+n])
            cand_ngrams[ngram] += 1
            
        # 计算匹配的n-gram数量（考虑重复）
        match_count = 0
        for ngram, count in cand_ngrams.items():
            # 取候选翻译中n-gram出现次数和参考翻译中出现次数的较小值
            match_count += min(count, ref_ngrams.get(ngram, 0))
            
        # 计算精度: 匹配的n-gram数量 / 候选翻译中n-gram总数
        precision = match_count / max(1, len(cand_tokens) - n + 1)
        precisions.append(precision)
    
    # 如果所有精度都为0，返回0
    if all(p == 0 for p in precisions):
        return 0
    
    # 计算几何平均数（所有精度的乘积开n次方）
    precision_product = 1
    for p in precisions:
        if p > 0:
            precision_product *= p
    
    precision_mean = precision_product ** (1 / len(precisions))
    
    # 计算简短惩罚项（防止生成过短翻译）
    # min(1, 候选长度/参考长度) - 如果候选比参考短，则惩罚
    brevity_penalty = min(1, len(cand_tokens) / max(1, len(ref_tokens)))
    
    # 计算最终BLEU分数 = 惩罚项 × 平均精度
    bleu = brevity_penalty * precision_mean
    
    return bleu

#=====================================================================================
# 1. 准备示例数据
#=====================================================================================
# 简单的中英对照句子 - 与RNN版本相同的数据集
# 在实际应用中，应使用更大的数据集来训练模型
chinese_sentences = [
    '你好',
    '谢谢',
    '我爱你',
    '再见',
    '早上好',
    '晚上好',
    '我很高兴认识你',
    '我叫小明',
    '今天天气很好',
    '明天见'
]  # 格式: 列表，每个元素是一个中文句子字符串

english_sentences = [
    'hello',
    'thank you',
    'i love you',
    'goodbye',
    'good morning',
    'good evening',
    'nice to meet you',
    'my name is xiaoming',
    'the weather is nice today',
    'see you tomorrow'
]  # 格式: 列表，每个元素是一个英文句子字符串

# 添加特殊标记: <sos>表示序列开始，<eos>表示序列结束
# 这些标记对于序列到序列模型至关重要
english_tokenized = [['<sos>'] + list(sent) + ['<eos>'] for sent in english_sentences]
# 格式: 列表的列表，每个内部列表是一个英文句子，分解为字符列表，并添加特殊标记

chinese_tokenized = [list(sent) for sent in chinese_sentences]
# 格式: 列表的列表，每个内部列表是一个中文句子，分解为字符列表

print("数据示例:")
for zh, en in zip(chinese_tokenized[:3], english_tokenized[:3]):
    print(f"中文: {zh}, 英文: {en}")

#=====================================================================================
# 2. 数据预处理
#=====================================================================================
# 构建词汇表函数 - 创建从单词到索引的映射和从索引到单词的映射
def build_vocab(texts):
    """
    构建词汇表并创建词到索引、索引到词的映射
    
    参数:
        texts: 列表的列表，每个内部列表包含一个句子的标记
        
    返回:
        vocab: 词汇表列表，包含所有唯一标记
        word2idx: 从词到索引的映射字典
        idx2word: 从索引到词的映射字典
    """
    # 收集所有词
    all_words = []  # 将所有句子的所有标记展平为一个列表
    for text in texts:
        all_words.extend(text)
    
    # 计算词频
    counter = Counter(all_words)  # {标记: 出现次数}
    
    # 构建词汇表，按频率排序
    # <pad>作为索引0，用于序列填充
    vocab = ['<pad>'] + [word for word, _ in counter.most_common()]
    # 格式: 列表，第一个元素是<pad>，后续是按频率排序的词
    
    # 构建映射
    word2idx = {word: idx for idx, word in enumerate(vocab)}  # {词: 索引}
    idx2word = {idx: word for word, idx in word2idx.items()}  # {索引: 词}
    
    return vocab, word2idx, idx2word

# 构建中英文词汇表
zh_vocab, zh_word2idx, zh_idx2word = build_vocab(chinese_tokenized)
# zh_vocab: 中文词汇表列表
# zh_word2idx: 中文词到索引的映射字典 {词: 索引}
# zh_idx2word: 中文索引到词的映射字典 {索引: 词}

en_vocab, en_word2idx, en_idx2word = build_vocab(english_tokenized)
# en_vocab: 英文词汇表列表
# en_word2idx: 英文词到索引的映射字典 {词: 索引}
# en_idx2word: 英文索引到词的映射字典 {索引: 词}

print(f"中文词汇量: {len(zh_vocab)}")
print(f"英文词汇量: {len(en_vocab)}")

# 将文本转换为索引序列
def text_to_indices(texts, word2idx):
    """
    将文本标记列表转换为对应的索引序列
    
    参数:
        texts: 列表的列表，每个内部列表是一个标记化的句子
        word2idx: 词到索引的映射字典
        
    返回:
        列表的列表，每个内部列表是句子中词的索引
    """
    return [[word2idx[word] for word in text] for text in texts]
    # 格式: 列表的列表，每个内部列表包含一个句子中所有词的索引

# 转换训练数据
zh_indices = text_to_indices(chinese_tokenized, zh_word2idx)  # 中文句子索引列表
en_indices = text_to_indices(english_tokenized, en_word2idx)  # 英文句子索引列表

# 填充序列 - 使所有序列具有相同长度，以便批处理
def pad_sequences(sequences, pad_idx=0):
    """
    将不等长的序列填充到相同长度
    
    参数:
        sequences: 列表的列表，每个内部列表是一个索引序列
        pad_idx: 用于填充的索引值，默认为0（对应<pad>标记）
        
    返回:
        列表的列表，每个内部列表是填充后的索引序列
    """
    max_len = max(len(seq) for seq in sequences)  # 找出最长序列的长度
    padded_seqs = []
    for seq in sequences:
        # 在序列末尾添加pad_idx，直到达到最大长度
        padded = seq + [pad_idx] * (max_len - len(seq))
        padded_seqs.append(padded)
    return padded_seqs
    # 格式: 列表的列表，每个内部列表是填充后的索引序列，所有内部列表长度相同

# 填充数据
zh_padded = pad_sequences(zh_indices)  # 填充后的中文索引序列
en_padded = pad_sequences(en_indices)  # 填充后的英文索引序列

# 转换为PyTorch张量，以便输入模型
zh_tensor = torch.tensor(zh_padded, dtype=torch.long)  # 形状: [batch_size, max_zh_len]
en_tensor = torch.tensor(en_padded, dtype=torch.long)  # 形状: [batch_size, max_en_len]

print(f"中文张量形状: {zh_tensor.shape}")
print(f"英文张量形状: {en_tensor.shape}")

#=====================================================================================
# 3. 定义Transformer模型
#=====================================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码模块：为序列中的每个位置提供唯一的表示
    
    Transformer没有循环或卷积，无法感知序列中的位置信息
    位置编码使用正弦和余弦函数，为序列的每个位置添加唯一的位置信息
    
    位置编码公式:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    其中pos是位置，i是维度
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        初始化位置编码模块
        
        参数:
            d_model (int): 嵌入维度
            dropout (float): dropout概率
            max_len (int): 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # dropout层，防止过拟合

        # 创建位置编码矩阵 - 形状: [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)  # 位置编码矩阵，初始为全0
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引: [max_len, 1]
        
        # 创建除数向量: [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 填充位置编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引使用cos
        
        pe = pe.unsqueeze(0)  # 添加批次维度: [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为持久缓冲区，不作为模型参数更新

    def forward(self, x):
        """
        前向传播，添加位置编码到输入嵌入
        
        参数:
            x: 词嵌入张量
               可以是[seq_len, batch_size, embed_dim]或[batch_size, seq_len, embed_dim]
               
        返回:
            应用了位置编码和dropout的张量，形状与输入相同
        """
        # 根据输入张量的形状决定如何添加位置编码
        if x.dim() == 3 and x.size(0) != self.pe.size(0):  # [seq_len, batch_size, embed_dim]
            x = x + self.pe.transpose(0, 1)[:x.size(0), :]
        else:  # [batch_size, seq_len, embed_dim]
            x = x + self.pe[:, :x.size(1), :]  # 将位置编码加到嵌入上
        return self.dropout(x)  # 应用dropout并返回结果

class Transformer(nn.Module):
    """
    Transformer翻译模型
    
    包含:
    1. 源语言和目标语言的嵌入层
    2. 位置编码
    3. 多层自注意力编码器
    4. 多层带掩码的自注意力解码器
    5. 最终的线性输出层
    """
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_heads, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, dropout, device, max_len=100):
        """
        初始化Transformer模型
        
        参数:
            src_vocab_size (int): 源语言词汇表大小
            trg_vocab_size (int): 目标语言词汇表大小
            d_model (int): 模型维度，嵌入和位置编码的维度
            n_heads (int): 注意力头数
            num_encoder_layers (int): 编码器层数
            num_decoder_layers (int): 解码器层数
            dim_feedforward (int): 前馈神经网络隐藏层维度
            dropout (float): dropout概率
            device (torch.device): 计算设备
            max_len (int): 最大序列长度
        """
        super().__init__()
        
        # 词嵌入层 - 将词索引转换为稠密向量表示
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)  # 源语言嵌入: [vocab_size, d_model]
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)  # 目标语言嵌入: [vocab_size, d_model]
        
        # 位置编码 - 添加位置信息
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer模型 - PyTorch内置的Transformer实现
        self.transformer = nn.Transformer(
            d_model=d_model,               # 模型维度
            nhead=n_heads,                 # 多头注意力中的头数
            num_encoder_layers=num_encoder_layers,  # 编码器层数
            num_decoder_layers=num_decoder_layers,  # 解码器层数
            dim_feedforward=dim_feedforward,        # 前馈网络维度
            dropout=dropout,               # dropout概率
            batch_first=True               # 输入张量的形状是[batch, seq, feature]
        )
        
        # 输出层 - 将Transformer输出映射回词汇表
        self.fc_out = nn.Linear(d_model, trg_vocab_size)  # [d_model, trg_vocab_size]
        
        # 其他属性
        self.d_model = d_model
        self.device = device
        
        # 初始化模型参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数，使用Xavier均匀初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 使权重服从均匀分布，保持输入输出方差一致
                
    def make_src_mask(self, src):
        """
        创建源序列的填充掩码
        
        参数:
            src: 源序列张量 [batch_size, src_len]
            
        返回:
            src_pad_mask: 源序列填充掩码 [batch_size, src_len]
                          值为True的位置将被屏蔽(对应填充标记)
        """
        # PyTorch 1.9及以上版本需要key_padding_mask形状为[batch_size, src_len]
        # 掩码中True表示该位置是填充位置，应该被屏蔽
        src_pad_mask = (src == 0)  # 假设0是<pad>标记的索引
        return src_pad_mask
    
    def make_trg_mask(self, trg):
        """
        创建目标序列的掩码
        
        参数:
            trg: 目标序列张量 [batch_size, trg_len]
            
        返回:
            trg_pad_mask: 目标序列填充掩码 [batch_size, trg_len]
            trg_sub_mask: 目标序列后续词掩码 [trg_len, trg_len]
                          用于确保预测时只能看到当前位置之前的词
        """
        # 填充掩码 [batch_size, trg_len] - 用于屏蔽填充标记
        trg_pad_mask = (trg == 0)
        
        # 后续词掩码 [trg_len, trg_len] - 用于解码器的自注意力
        # 这是一个上三角矩阵，屏蔽未来位置的标记
        trg_len = trg.shape[1]
        # torch.triu创建上三角矩阵，diagonal=1表示主对角线上移一位
        # 结果是一个布尔掩码，True表示应该被屏蔽的位置
        trg_sub_mask = torch.triu(torch.ones((trg_len, trg_len), device=self.device), diagonal=1).bool()
        
        return trg_pad_mask, trg_sub_mask
    
    def forward(self, src, trg):
        """
        前向传播
        
        参数:
            src: 源序列张量 [batch_size, src_len]
            trg: 目标序列张量 [batch_size, trg_len]
            
        返回:
            output: 模型输出 [batch_size, trg_len, trg_vocab_size]
                    代表每个位置每个词的概率分布
        """
        # 创建源序列和目标序列的掩码
        src_pad_mask = self.make_src_mask(src)  # [batch_size, src_len]
        trg_pad_mask, trg_sub_mask = self.make_trg_mask(trg)  # [batch_size, trg_len], [trg_len, trg_len]
        
        # 嵌入和位置编码
        # 1. 将词索引转换为嵌入向量
        # 2. 乘以sqrt(d_model)以保持嵌入和位置编码的相对强度
        # 3. 添加位置编码
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)  # [batch_size, src_len, d_model]
        # 转置是为了适应PositionalEncoding的输入格式，然后再转置回来
        src_embedded = self.pos_encoder(src_embedded.transpose(0, 1)).transpose(0, 1)
        
        trg_embedded = self.trg_embedding(trg) * math.sqrt(self.d_model)  # [batch_size, trg_len, d_model]
        trg_embedded = self.pos_encoder(trg_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Transformer模型的前向传播
        # src_key_padding_mask: 源序列填充掩码，指示哪些位置是填充标记
        # tgt_key_padding_mask: 目标序列填充掩码
        # memory_key_padding_mask: 用于注意力计算的记忆填充掩码，通常与src_key_padding_mask相同
        # tgt_mask: 目标序列的自注意力掩码，确保位置i只能注意到位置0到i-1
        output = self.transformer(
            src=src_embedded,              # [batch_size, src_len, d_model]
            tgt=trg_embedded,              # [batch_size, trg_len, d_model]
            src_key_padding_mask=src_pad_mask,        # [batch_size, src_len]
            tgt_key_padding_mask=trg_pad_mask,        # [batch_size, trg_len]
            memory_key_padding_mask=src_pad_mask,     # [batch_size, src_len]
            tgt_mask=trg_sub_mask                     # [trg_len, trg_len]
        )  # 输出: [batch_size, trg_len, d_model]
        
        # 线性层将Transformer输出映射到词汇表大小
        output = self.fc_out(output)  # [batch_size, trg_len, trg_vocab_size]
        
        return output

#=====================================================================================
# 4. 模型参数与初始化
#=====================================================================================
# 超参数
INPUT_DIM = len(zh_vocab)  # 源语言词汇表大小（中文）
OUTPUT_DIM = len(en_vocab)  # 目标语言词汇表大小（英文）
D_MODEL = 256    # 模型维度/嵌入维度，Transformer内部所有子层输出维度
N_HEADS = 8      # 多头注意力中的头数，必须能被D_MODEL整除
N_ENCODER_LAYERS = 3  # 编码器层数，每层包含自注意力和前馈网络
N_DECODER_LAYERS = 3  # 解码器层数，每层包含自注意力、编码器-解码器注意力和前馈网络
DIM_FEEDFORWARD = 512  # 前馈网络隐藏层维度，通常是D_MODEL的2-4倍
DROPOUT = 0.1    # Dropout概率，用于防止过拟合
BATCH_SIZE = 10  # 批量大小，这里使用全部数据作为一个批次
N_EPOCHS = 200   # 训练轮数，模型将遍历整个数据集的次数
CLIP = 1.0       # 梯度裁剪阈值，防止梯度爆炸
PRINT_EVERY = 20  # 打印频率，每隔多少个epoch打印一次训练信息

# 使用CPU还是GPU，基于是否有可用的CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型实例
model = Transformer(
    src_vocab_size=INPUT_DIM,       # 源语言词汇表大小
    trg_vocab_size=OUTPUT_DIM,      # 目标语言词汇表大小
    d_model=D_MODEL,                # 模型维度
    n_heads=N_HEADS,                # 注意力头数
    num_encoder_layers=N_ENCODER_LAYERS,  # 编码器层数
    num_decoder_layers=N_DECODER_LAYERS,  # 解码器层数
    dim_feedforward=DIM_FEEDFORWARD,      # 前馈网络维度
    dropout=DROPOUT,                # dropout概率
    device=device                   # 计算设备
).to(device)  # 将模型移动到指定设备（CPU/GPU）

# 计算模型参数数量，用于了解模型复杂度
def count_parameters(model):
    """
    计算模型的可训练参数数量
    
    参数:
        model: PyTorch模型
        
    返回:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'模型参数数量: {count_parameters(model):,}')

# 定义优化器 - Adam优化器是Transformer的标准选择
# 使用较小的学习率0.0001，因为数据集小，防止模型学习过快
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 定义损失函数 - 交叉熵损失，忽略填充标记（<pad>索引为0）
# 忽略填充标记很重要，否则模型会浪费精力预测填充位置
PAD_IDX = en_word2idx['<pad>']  # 填充标记的索引
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

#=====================================================================================
# 5. 训练与评估函数
#=====================================================================================
def train(model, iterator, optimizer, criterion, clip):
    """
    训练函数 - 对模型进行一个epoch的训练
    
    参数:
        model: Transformer模型
        iterator: 数据迭代器，提供(src, trg)批次
        optimizer: 优化器，如Adam
        criterion: 损失函数，如交叉熵
        clip: 梯度裁剪阈值，防止梯度爆炸
        
    返回:
        epoch_loss: 整个epoch的平均损失
    """
    model.train()  # 设置模型为训练模式，启用dropout等
    epoch_loss = 0  # 累计整个epoch的损失
    
    for batch in iterator:
        src, trg = batch  # src: [batch_size, src_len], trg: [batch_size, trg_len]
        src, trg = src.to(device), trg.to(device)  # 将数据移至训练设备(CPU/GPU)
        
        optimizer.zero_grad()  # 清除之前的梯度
        
        # trg 去掉最后一个token以用于输入，因为我们预测下一个token
        # 例如，如果目标是 [<sos>, a, b, c, <eos>]
        # 我们的输入是 [<sos>, a, b, c]，目标输出是 [a, b, c, <eos>]
        output = model(src, trg[:, :-1])  # [batch_size, trg_len-1, trg_vocab_size]
        
        # 将输出重塑为 [batch_size * (trg_len-1), output_dim]
        # 这是为了适应CrossEntropyLoss的输入要求
        output_dim = output.shape[-1]  # 输出维度，等于目标词汇表大小
        output = output.reshape(-1, output_dim)  # [batch_size * (trg_len-1), output_dim]
        
        # 将目标去掉第一个token(sos)以用于计算损失
        # 使目标与输出对齐：输入[<sos>, a, b, c]，输出[a, b, c, <eos>]
        trg = trg[:, 1:].reshape(-1)  # [batch_size * (trg_len-1)]
        
        # 计算损失
        loss = criterion(output, trg)  # 交叉熵损失
        loss.backward()  # 反向传播，计算梯度
        
        # 梯度裁剪，防止梯度爆炸
        # 将所有参数的梯度范数裁剪到指定阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # 更新参数
        optimizer.step()
        
        epoch_loss += loss.item()  # 累加批次损失
    
    return epoch_loss / len(iterator)  # 返回平均损失

def evaluate(model, iterator, criterion):
    """
    评估函数 - 在不更新参数的情况下评估模型性能
    
    参数:
        model: Transformer模型
        iterator: 数据迭代器，提供(src, trg)批次
        criterion: 损失函数，如交叉熵
        
    返回:
        epoch_loss: 整个评估集的平均损失
    """
    model.eval()  # 设置模型为评估模式，禁用dropout等
    epoch_loss = 0
    
    with torch.no_grad():  # 不计算梯度，节省内存
        for batch in iterator:
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            
            # trg 去掉最后一个token以用于输入
            output = model(src, trg[:, :-1])  # [batch_size, trg_len-1, trg_vocab_size]
            
            # 将输出重塑为 [batch_size * (trg_len-1), output_dim]
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            
            # 将目标去掉第一个token(sos)以用于计算损失
            trg = trg[:, 1:].reshape(-1)
            
            # 计算损失
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)  # 返回平均损失

def translate_sentence(model, sentence, src_field, trg_field, device, max_len=50):
    """
    翻译单个句子的函数
    
    参数:
        model: 训练好的Transformer模型
        sentence: 要翻译的源语言句子，可以是字符串或标记列表
        src_field: 源语言词到索引的映射字典
        trg_field: 目标语言索引到词的映射字典
        device: 计算设备
        max_len: 生成的最大序列长度
        
    返回:
        翻译后的标记列表
    """
    model.eval()  # 设置模型为评估模式
    
    # 处理输入句子，确保是标记列表
    if isinstance(sentence, str):
        tokens = list(sentence)  # 将字符串转换为字符列表
    else:
        tokens = sentence  # 已经是标记列表
    
    # 将源语言标记转换为索引
    src_indices = [src_field.get(token, 0) for token in tokens]  # 未知词使用索引0(<pad>)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    
    # 添加调试信息
    print(f"\n翻译句子: {''.join(tokens)}")
    print(f"源语言索引: {src_indices}")
    
    # 创建一个包含起始标记的目标张量
    # 在词汇表中，索引1通常对应<sos>标记
    trg_indices = [1]  # 使用索引1作为起始标记
    trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)  # [1, 1]
    
    eos_index = 2  # 假设索引2是<eos>标记
    
    # 自回归生成：逐步预测下一个词
    for i in range(max_len):
        # 获取模型预测
        with torch.no_grad():  # 不计算梯度
            output = model(src_tensor, trg_tensor)  # [1, trg_len, trg_vocab_size]
        
        # 获取最后一个时间步的预测（最可能的下一个词）
        pred_token = output[:, -1, :].argmax(1).item()  # 最后一个位置最高概率的词索引
        
        # 添加调试信息
        pred_word = trg_field.get(pred_token, '<未知>')
        print(f"  步骤 {i+1}: 预测词索引 {pred_token} -> 词 '{pred_word}'")
        
        # 添加预测的token到目标序列
        trg_indices.append(pred_token)
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)  # [1, trg_len+1]
        
        # 如果预测到结束标记或padding标记，停止预测
        if pred_token == eos_index or pred_token == 0:
            print(f"  在步骤 {i+1} 结束预测，遇到了{'<eos>' if pred_token == eos_index else '<pad>'}")
            break
    
    # 将索引转换回词
    trg_tokens = [trg_field.get(i, '') for i in trg_indices]
    print(f"最终翻译的标记: {trg_tokens}")
    print(f"最终翻译: {''.join(trg_tokens[1:])}")
    
    # 去掉起始标记
    return trg_tokens[1:]  # 返回不含起始标记的翻译结果

#=====================================================================================
# 6. 训练过程
#=====================================================================================
# 准备训练数据 - 简单批处理
def create_batches(data):
    """
    创建数据批次
    
    参数:
        data: 包含源语言和目标语言张量的元组 (src_tensor, trg_tensor)
        
    返回:
        列表，包含(src, trg)批次
    """
    return [(data[0], data[1])]  # 对于小数据集，将所有数据作为一个批次

# 创建数据批次
train_iterator = create_batches((zh_tensor, en_tensor))

# 训练模型
print("开始训练...")
train_losses = []  # 记录每个epoch的训练损失
eval_losses = []   # 记录每个epoch的评估损失
best_valid_loss = float('inf')  # 跟踪最佳评估损失

# 训练循环
for epoch in range(N_EPOCHS):
    start_time = time.time()  # 记录开始时间
    
    # 训练一个epoch
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    # 评估模型
    eval_loss = evaluate(model, train_iterator, criterion)
    
    # 计算耗时
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    # 记录损失
    train_losses.append(train_loss)
    eval_losses.append(eval_loss)
    
    # 记录最佳模型
    if eval_loss < best_valid_loss:
        best_valid_loss = eval_loss
        print(f'Epoch {epoch+1:03}/{N_EPOCHS} - 发现新的最佳模型，评估损失: {eval_loss:.6f}')
    
    # 每个epoch都打印训练信息
    if (epoch + 1) % PRINT_EVERY == 0:
        print(f'Epoch: {epoch+1:03}/{N_EPOCHS} | 训练损失: {train_loss:.4f} | 评估损失: {eval_loss:.4f} | 用时: {epoch_mins}m {epoch_secs:.0f}s')
    else:
        # 每个epoch都打印简略信息
        print(f'Epoch: {epoch+1:03}/{N_EPOCHS} | 损失: {train_loss:.6f}')

print(f"训练完成。最终训练损失: {train_loss:.4f}, 评估损失: {eval_loss:.4f}")

#=====================================================================================
# 7. 绘制损失曲线
#=====================================================================================
plt.figure(figsize=(10, 6))  # 创建一个图形，设置大小
plt.plot(train_losses, label='训练损失')  # 绘制训练损失曲线
plt.plot(eval_losses, label='评估损失')   # 绘制评估损失曲线
plt.xlabel('Epoch')  # X轴标签
plt.ylabel('损失')    # Y轴标签
plt.title('训练和评估损失')  # 图标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.savefig('transformer_loss_curve.png')  # 保存图片到文件
plt.close()  # 关闭图形

#=====================================================================================
# 8. 测试翻译效果
#=====================================================================================
print("\n===== 测试翻译结果 =====")
translations = []  # 存储所有翻译结果

# 遍历测试集中的每个句子
for i, sentence in enumerate(chinese_tokenized):
    # 翻译当前句子
    translation = translate_sentence(model, sentence, zh_word2idx, en_idx2word, device)
    
    # 清理翻译结果，去掉空字符和<eos>标记
    translation = [token for token in translation if token and token != '<eos>']
    
    # 将字符连接成单词
    translation_text = ''.join(translation)  # 连接字符，形成完整翻译
    translations.append(translation_text)    # 添加到翻译结果列表
    
    # 打印结果，对比源句子、翻译结果和参考翻译
    print(f'中文: {"".join(sentence)}')
    print(f'翻译: {translation_text}')
    print(f'参考: {english_sentences[i]}')
    print('-' * 50)  # 分隔线

# 计算整体BLEU分数
# 对每个句子对计算BLEU分数，然后取平均值
avg_bleu = compute_bleu(' '.join(english_sentences[0]), ' '.join(translations[0]))
for i in range(1, len(english_sentences)):
    avg_bleu += compute_bleu(' '.join(english_sentences[i]), ' '.join(translations[i]))
avg_bleu /= len(english_sentences)  # 计算平均BLEU分数

print(f"\n整体翻译BLEU分数: {avg_bleu:.4f}")

# 对特定句子评估翻译质量
perfect_match_count = 0  # 完全匹配的句子数量
for i, (trans, ref) in enumerate(zip(translations, english_sentences)):
    # 计算单个句子的BLEU分数
    bleu = compute_bleu(' '.join(ref), ' '.join(trans))
    
    # 检查是否完全匹配
    if trans == ref:
        perfect_match_count += 1
        
    # 打印句子级别的评估结果    
    print(f"句子 {i+1} BLEU分数: {bleu:.4f} - '{chinese_sentences[i]}' -> '{trans}'")

# 打印总体匹配统计
print(f"\n完全匹配的句子数: {perfect_match_count}/{len(english_sentences)}")
print(f"匹配率: {perfect_match_count/len(english_sentences)*100:.2f}%")

#=====================================================================================
# 9. 比较与RNN版本的性能差异
#=====================================================================================
print("\n===== Transformer vs. RNN性能比较 =====")
print("Transformer优势:")
print("1. 并行计算：所有位置同时计算，而不是顺序依赖")
print("2. 长距离依赖：注意力机制可以直接建立远距离连接")
print("3. 多头注意力：可以同时关注不同的特征表示")
print("4. 位置编码：提供序列顺序信息")

print("\n对于本实验中的小数据集:")
print("- Transformer可能过度参数化，容易过拟合")
print("- 参数数量更多，计算更复杂")
print("- 但在处理长句子时仍可能有优势")
print("- 实际应用中通常在大数据集上表现更好") 