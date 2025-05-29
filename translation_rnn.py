"""
简单的RNN中英翻译模型示例
此代码展示了如何使用RNN（LSTM）构建一个简单的中英翻译系统
这是一个序列到序列(Seq2Seq)模型的实现，它由编码器和解码器两部分组成
"""

# 导入必要的库
import numpy as np                      # 用于数值计算和数组操作
import torch                            # PyTorch深度学习框架
import torch.nn as nn                   # 神经网络模块
import torch.optim as optim             # 优化器
import random                           # 随机数生成
import time                             # 计时
from collections import Counter         # 用于构建词汇表
import re                               # 用于文本处理

# 设置随机种子，确保结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 简化版BLEU评分函数
def compute_bleu(reference, candidate, max_n=4):
    """
    计算BLEU分数
    
    参数:
        reference: 参考翻译（字符串）
        candidate: 候选翻译（字符串）
        max_n: 最大n-gram
        
    返回:
        BLEU分数
    """
    # 预处理
    reference = reference.lower()
    candidate = candidate.lower()
    
    # 分词（为简化起见，我们按字符分割）
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    # 如果候选翻译为空，返回0
    if len(cand_tokens) == 0:
        return 0
    
    # 计算n-gram精度
    precisions = []
    for n in range(1, min(max_n, len(cand_tokens)) + 1):
        # 创建n-gram
        ref_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = ' '.join(ref_tokens[i:i+n])
            ref_ngrams[ngram] += 1
            
        cand_ngrams = Counter()
        for i in range(len(cand_tokens) - n + 1):
            ngram = ' '.join(cand_tokens[i:i+n])
            cand_ngrams[ngram] += 1
            
        # 计算匹配的n-gram数量
        match_count = 0
        for ngram, count in cand_ngrams.items():
            match_count += min(count, ref_ngrams.get(ngram, 0))
            
        # 计算精度
        precision = match_count / max(1, len(cand_tokens) - n + 1)
        precisions.append(precision)
    
    # 如果所有精度都为0，返回0
    if all(p == 0 for p in precisions):
        return 0
    
    # 计算几何平均数
    precision_product = 1
    for p in precisions:
        if p > 0:
            precision_product *= p
    
    precision_mean = precision_product ** (1 / len(precisions))
    
    # 计算简单的惩罚项（不是标准BLEU的惩罚）
    brevity_penalty = min(1, len(cand_tokens) / max(1, len(ref_tokens)))
    
    # 计算最终BLEU分数
    bleu = brevity_penalty * precision_mean
    
    return bleu

#=====================================================================================
# 1. 准备示例数据（实际应用中应使用更大的数据集）
#=====================================================================================
# 简单的中英对照句子 - 这是我们的平行语料库，每个中文句子对应一个英文翻译
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
]

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
]

# 添加起始标记和结束标记
english_tokenized = [['<sos>'] + list(sent) + ['<eos>'] for sent in english_sentences]
chinese_tokenized = [list(sent) for sent in chinese_sentences]

print("数据示例:")
for zh, en in zip(chinese_tokenized[:3], english_tokenized[:3]):
    print(f"中文: {zh}, 英文: {en}")

#=====================================================================================
# 2. 数据预处理 - 将文本转换为神经网络可以处理的数值形式
#=====================================================================================
# 构建词汇表函数
def build_vocab(texts):
    # 收集所有词
    all_words = []
    for text in texts:
        all_words.extend(text)
    
    # 计算词频
    counter = Counter(all_words)
    
    # 构建词汇表，按频率排序
    vocab = ['<pad>'] + [word for word, _ in counter.most_common()]
    
    # 构建映射
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return vocab, word2idx, idx2word

# 构建中英文词汇表
zh_vocab, zh_word2idx, zh_idx2word = build_vocab(chinese_tokenized)
en_vocab, en_word2idx, en_idx2word = build_vocab(english_tokenized)

print(f"中文词汇量: {len(zh_vocab)}")
print(f"英文词汇量: {len(en_vocab)}")

# 将文本转换为索引序列
def text_to_indices(texts, word2idx):
    return [[word2idx[word] for word in text] for text in texts]

# 转换训练数据
zh_indices = text_to_indices(chinese_tokenized, zh_word2idx)
en_indices = text_to_indices(english_tokenized, en_word2idx)

# 填充序列
def pad_sequences(sequences, pad_idx=0):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        padded = seq + [pad_idx] * (max_len - len(seq))
        padded_seqs.append(padded)
    return padded_seqs

# 填充数据
zh_padded = pad_sequences(zh_indices)
en_padded = pad_sequences(en_indices)

# 转换为PyTorch张量
zh_tensor = torch.tensor(zh_padded, dtype=torch.long)
en_tensor = torch.tensor(en_padded, dtype=torch.long)

print(f"中文张量形状: {zh_tensor.shape}")
print(f"英文张量形状: {en_tensor.shape}")

#=====================================================================================
# 3. 构建模型 - 编码器-解码器架构
#=====================================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim]
        # hidden = [1, batch_size, hid_dim]
        # cell = [1, batch_size, hid_dim]
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input = [batch_size]
        # hidden = [1, batch_size, hid_dim]
        # cell = [1, batch_size, hid_dim]
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # 用于存储预测结果
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 编码器前向传播
        hidden, cell = self.encoder(src)
        
        # 第一个输入是起始标记
        input = trg[0, :]
        
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # 保存预测结果
            outputs[t] = output
            
            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 获取最可能的预测词
            top1 = output.argmax(1)
            
            # 如果使用教师强制，使用真实目标；否则使用预测结果
            input = trg[t] if teacher_force else top1
            
        return outputs

#=====================================================================================
# 4. 初始化模型
#=====================================================================================
# 设置超参数
INPUT_DIM = len(zh_vocab)
OUTPUT_DIM = len(en_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# 使用CPU还是GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型实例
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# 初始化模型权重
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

# 计算参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'模型参数数量: {count_parameters(model):,}')

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=en_word2idx['<pad>'])

#=====================================================================================
# 5. 训练模型
#=====================================================================================
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg = [trg_len, batch_size]
        # output = [trg_len, batch_size, output_dim]
        
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)
        
        # trg = [(trg_len-1) * batch_size]
        # output = [(trg_len-1) * batch_size, output_dim]
        
        loss = criterion(output, trg)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

#=====================================================================================
# 6. 评估模型
#=====================================================================================
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            src = batch[0].to(device)
            trg = batch[1].to(device)
            
            output = model(src, trg, 0)  # 测试时不使用教师强制
            
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# 计算翻译的BLEU分数
def evaluate_translations(model, sentences, references, src_field, trg_field, device):
    model.eval()
    bleu_scores = []
    
    for i, sentence in enumerate(sentences):
        # 翻译
        translation_tokens = translate_sentence(sentence, src_field, trg_field, model, device)
        
        # 清理翻译结果，去掉空字符和<eos>
        translation_tokens = [token for token in translation_tokens if token and token != '<eos>']
        translation = ''.join(translation_tokens)
        
        # 为BLEU评分分词（这里简单地将字符作为单词处理）
        translation_words = ' '.join(translation)
        reference_words = ' '.join(references[i])
        
        # 计算BLEU分数
        bleu = compute_bleu(reference_words, translation_words)
        bleu_scores.append(bleu)
        
    # 返回平均BLEU分数
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

#=====================================================================================
# 7. 翻译函数
#=====================================================================================
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    
    if isinstance(sentence, str):
        tokens = list(sentence)
    else:
        tokens = sentence
        
    # 转换为索引
    src_indices = [src_field.get(token, 0) for token in tokens]
    
    # 转换为张量
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)
    
    # 编码
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    
    # 首个token是<sos>对应的索引，我们使用字典中第0个词
    # 在构建词典时，<pad>为索引0，我们使用索引1（通常是<sos>）
    trg_indices = [1]  # 起始标记通常是索引1
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        
        # 如果预测到索引0（通常是<pad>或<eos>），结束翻译
        if pred_token == 0:
            break
        
        trg_indices.append(pred_token)
    
    # 转换为词
    trg_tokens = [trg_field.get(i, '') for i in trg_indices[1:]]  # 跳过起始标记
    
    return trg_tokens

#=====================================================================================
# 8. 训练过程
#=====================================================================================
# 准备训练数据
# 将数据转置为(seq_len, batch_size)格式
zh_tensor = zh_tensor.permute(1, 0)
en_tensor = en_tensor.permute(1, 0)

# 创建一个简单的批处理器
def create_batches(data, batch_size=1):
    return [(data[0], data[1])]  # 对于小数据集，我们将所有数据作为一个批次

# 创建数据批次
train_iterator = create_batches((zh_tensor, en_tensor))

# 训练参数
N_EPOCHS = 300
CLIP = 1.0
PRINT_EVERY = 20

# 开始训练
print("开始训练...")
best_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    
    # 使用evaluate函数评估模型在训练集上的性能
    eval_loss = evaluate(model, train_iterator, criterion)
    
    end_time = time.time()
    
    if (epoch + 1) % PRINT_EVERY == 0:
        print(f'Epoch: {epoch+1:03} | 训练损失: {train_loss:.3f} | 评估损失: {eval_loss:.3f} | 耗时: {end_time - start_time:.3f}秒')
    
    # 保存最佳模型
    if eval_loss < best_loss:
        best_loss = eval_loss

print(f"训练完成。最终训练损失: {train_loss:.4f}, 评估损失: {eval_loss:.4f}")

#=====================================================================================
# 9. 测试翻译效果
#=====================================================================================
print("\n===== 测试翻译结果 =====")
translations = []

for i, sentence in enumerate(chinese_tokenized):
    # 翻译
    translation = translate_sentence(sentence, zh_word2idx, en_idx2word, model, device)
    
    # 清理翻译结果，去掉空字符和<eos>
    translation = [token for token in translation if token and token != '<eos>']
    
    # 将字符连接成单词
    translation_text = ''.join(translation)
    translations.append(translation_text)
    
    # 打印结果
    print(f'中文: {"".join(sentence)}')
    print(f'翻译: {translation_text}')
    print(f'参考: {english_sentences[i]}')
    print('-' * 50)

# 计算整体BLEU分数
avg_bleu = compute_bleu(' '.join(english_sentences[0]), ' '.join(translations[0]))
for i in range(1, len(english_sentences)):
    avg_bleu += compute_bleu(' '.join(english_sentences[i]), ' '.join(translations[i]))
avg_bleu /= len(english_sentences)

print(f"\n整体翻译BLEU分数: {avg_bleu:.4f}")

# 对特定句子评估翻译质量
perfect_match_count = 0
for i, (trans, ref) in enumerate(zip(translations, english_sentences)):
    bleu = compute_bleu(' '.join(ref), ' '.join(trans))
    if trans == ref:
        perfect_match_count += 1
    print(f"句子 {i+1} BLEU分数: {bleu:.4f} - '{chinese_sentences[i]}' -> '{trans}'")

print(f"\n完全匹配的句子数: {perfect_match_count}/{len(english_sentences)}")
print(f"匹配率: {perfect_match_count/len(english_sentences)*100:.2f}%") 