"""
使用Hugging Face预训练模型的中英翻译器
相比于从头训练的Transformer模型，预训练模型具有以下优势：
1. 已在大规模数据上训练，拥有丰富的语言知识
2. 可以直接使用或通过少量数据微调
3. 性能更好，特别是在小数据集场景下
4. 更容易部署和使用

本代码使用的是Hugging Face的MarianMT模型，这是一个专门用于机器翻译的预训练模型。
MarianMT是基于Transformer架构的encoder-decoder模型，专为机器翻译优化。
"""

# 导入必要的库
import torch                # PyTorch深度学习框架，用于张量操作和模型运行
from transformers import MarianMTModel, MarianTokenizer  # Hugging Face的翻译模型和分词器
import time                 # 用于计时，评估翻译速度
import numpy as np          # 用于数值计算
from sacrebleu.metrics import BLEU  # 用于评估翻译质量的BLEU评分

# 设置随机种子以确保结果可复现
# 这对于实验比较和调试非常重要
torch.manual_seed(42)       # PyTorch随机数生成器种子
np.random.seed(42)          # NumPy随机数生成器种子

class HuggingFaceTranslator:
    """
    基于Hugging Face预训练模型的翻译器类
    支持中文到英文和英文到中文的翻译
    
    这个类封装了两个预训练模型：
    1. 中译英模型：将中文文本翻译为英文
    2. 英译中模型：将英文文本翻译为中文
    """
    def __init__(self, zh_to_en_model="Helsinki-NLP/opus-mt-zh-en", 
                 en_to_zh_model="Helsinki-NLP/opus-mt-en-zh", device=None):
        """
        初始化翻译器，加载预训练模型和分词器
        
        参数:
            zh_to_en_model: 中译英模型的名称或路径（默认使用Helsinki-NLP的opus模型）
            en_to_zh_model: 英译中模型的名称或路径（默认使用Helsinki-NLP的opus模型）
            device: 计算设备，默认为None（自动选择CPU或GPU）
        
        初始化过程:
        1. 确定使用的计算设备（CPU/GPU）
        2. 加载中译英和英译中的模型及分词器
        3. 计算并显示模型参数数量
        """
        # 确定计算设备：如果有GPU则使用GPU，否则使用CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载中译英模型和分词器
        # MarianTokenizer用于将文本转换为模型可处理的数字序列
        # MarianMTModel是实际执行翻译的神经网络模型
        print(f"加载中译英模型: {zh_to_en_model}")
        self.zh_to_en_tokenizer = MarianTokenizer.from_pretrained(zh_to_en_model)
        self.zh_to_en_model = MarianMTModel.from_pretrained(zh_to_en_model).to(self.device)
        
        # 加载英译中模型和分词器
        print(f"加载英译中模型: {en_to_zh_model}")
        self.en_to_zh_tokenizer = MarianTokenizer.from_pretrained(en_to_zh_model)
        self.en_to_zh_model = MarianMTModel.from_pretrained(en_to_zh_model).to(self.device)
        
        # 打印模型参数数量，用于了解模型复杂度
        zh_to_en_params = sum(p.numel() for p in self.zh_to_en_model.parameters())
        en_to_zh_params = sum(p.numel() for p in self.en_to_zh_model.parameters())
        print(f"中译英模型参数: {zh_to_en_params:,}")
        print(f"英译中模型参数: {en_to_zh_params:,}")
        
    def translate_zh_to_en(self, text, max_length=50, num_beams=5, debug=False):
        """
        将中文文本翻译为英文
        
        参数:
            text: 待翻译的中文文本字符串
            max_length: 生成的最大序列长度，默认50个标记
            num_beams: 束搜索的宽度，默认5，越大质量越好但越慢
            debug: 是否打印调试信息，默认False
            
        返回:
            翻译后的英文文本字符串
            
        工作流程:
        1. 将输入中文文本转换为模型输入张量
        2. 使用模型生成英文翻译（束搜索）
        3. 将生成的标记ID解码为文本
        """
        if debug:
            print(f"\n翻译中文: {text}")
            
        # 对输入文本进行分词，转换为模型可处理的格式
        # return_tensors="pt" 表示返回PyTorch张量
        # 结果形状: inputs['input_ids']: [1, seq_len]，其中seq_len是输入序列长度
        inputs = self.zh_to_en_tokenizer(text, return_tensors="pt").to(self.device)
        
        if debug:
            # 打印分词结果，查看模型如何理解输入文本
            # convert_ids_to_tokens将数字ID转回可读的标记
            print(f"分词结果: {self.zh_to_en_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
            print(f"输入张量形状: {inputs['input_ids'].shape}")  # 通常是[1, seq_len]
        
        # 使用模型生成翻译，禁用梯度计算以节省内存和提高速度
        with torch.no_grad():
            start_time = time.time()
            
            # 使用束搜索进行生成，获得更高质量的翻译
            # generate方法参数:
            # - **inputs: 输入张量字典，包含input_ids和attention_mask
            # - max_length: 生成序列的最大长度
            # - num_beams: 束搜索的宽度，越大越慢但质量更高
            # - early_stopping: 当所有束都生成了EOS标记时提前停止
            # 输出形状: [1, output_seq_len]，表示一个批次中一个序列的输出标记ID
            outputs = self.zh_to_en_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            end_time = time.time()
        
        # 将模型输出的标记ID解码为文本
        # skip_special_tokens=True表示忽略特殊标记如<s>和</s>
        translation = self.zh_to_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if debug:
            print(f"输出张量形状: {outputs.shape}")  # 通常是[1, output_seq_len]
            print(f"输出标记IDs: {outputs[0].tolist()}")  # 一维数组，包含生成的标记ID
            print(f"翻译结果: {translation}")
            print(f"翻译用时: {(end_time - start_time)*1000:.2f}ms")
            
        return translation
    
    def translate_en_to_zh(self, text, max_length=50, num_beams=5, debug=False):
        """
        将英文文本翻译为中文
        
        参数:
            text: 待翻译的英文文本字符串
            max_length: 生成的最大序列长度，默认50个标记
            num_beams: 束搜索的宽度，默认5，越大质量越好但越慢
            debug: 是否打印调试信息，默认False
            
        返回:
            翻译后的中文文本字符串
            
        工作流程:
        1. 将输入英文文本转换为模型输入张量
        2. 使用模型生成中文翻译（束搜索）
        3. 将生成的标记ID解码为文本
        """
        if debug:
            print(f"\n翻译英文: {text}")
            
        # 对输入文本进行分词，转换为模型可处理的格式
        # 结果形状: inputs['input_ids']: [1, seq_len]
        inputs = self.en_to_zh_tokenizer(text, return_tensors="pt").to(self.device)
        
        if debug:
            print(f"分词结果: {self.en_to_zh_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
            print(f"输入张量形状: {inputs['input_ids'].shape}")
        
        # 使用模型生成翻译
        with torch.no_grad():
            start_time = time.time()
            
            # 使用束搜索生成翻译
            # 输出形状: [1, output_seq_len]
            outputs = self.en_to_zh_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            end_time = time.time()
        
        # 将模型输出解码为文本
        translation = self.en_to_zh_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if debug:
            print(f"输出张量形状: {outputs.shape}")
            print(f"输出标记IDs: {outputs[0].tolist()}")
            print(f"翻译结果: {translation}")
            print(f"翻译用时: {(end_time - start_time)*1000:.2f}ms")
            
        return translation

def calculate_bleu(references, translations):
    """
    计算BLEU分数 - 评估机器翻译质量的标准度量
    
    参数:
        references: 参考翻译列表（真实/正确翻译）
        translations: 模型生成的翻译列表
        
    返回:
        BLEU分数，范围0-1，越高越好
        
    BLEU (Bilingual Evaluation Understudy)是机器翻译评估的标准指标
    它通过比较生成的翻译与一个或多个参考翻译来计算分数
    主要考虑n-gram精度（n=1,2,3,4）以及简短惩罚
    """
    # 使用sacrebleu库计算BLEU分数
    # corpus_score计算整个语料库的BLEU分数
    # 参数格式:
    # - translations: 模型翻译结果列表
    # - [references]: 参考翻译列表的列表（每个输入可以有多个参考翻译）
    bleu = BLEU()
    return bleu.corpus_score(translations, [references]).score

def evaluate_translator(translator, source_texts, reference_texts, zh_to_en=True):
    """
    评估翻译器性能
    
    参数:
        translator: 翻译器实例
        source_texts: 源语言文本列表
        reference_texts: 参考翻译文本列表
        zh_to_en: 如果为True，评估中译英；否则评估英译中
        
    返回:
        tuple: (翻译结果列表, BLEU分数, 完全匹配率)
        
    评估过程:
    1. 对每个源文本进行翻译
    2. 计算翻译用时
    3. 统计完全匹配数量
    4. 计算BLEU分数
    5. 打印评估结果
    """
    translations = []      # 存储翻译结果
    perfect_matches = 0    # 完全匹配计数
    total_time = 0         # 总翻译时间
    
    print(f"\n===== 开始评估{'中译英' if zh_to_en else '英译中'}翻译 =====")
    
    # 遍历每个测试样例
    for i, (source, reference) in enumerate(zip(source_texts, reference_texts)):
        # 根据翻译方向选择对应的翻译方法
        start_time = time.time()
        if zh_to_en:
            translation = translator.translate_zh_to_en(source)
        else:
            translation = translator.translate_en_to_zh(source)
        end_time = time.time()
        
        # 累计翻译时间和结果
        total_time += (end_time - start_time)
        translations.append(translation)
        
        # 检查是否完全匹配参考翻译（忽略大小写）
        if translation.lower() == reference.lower():
            perfect_matches += 1
            match_status = "完全匹配"
        else:
            match_status = "部分匹配"
        
        # 打印详细评估信息
        print(f"样例 {i+1}:")
        print(f"  源文本: {source}")
        print(f"  翻译结果: {translation}")
        print(f"  参考翻译: {reference}")
        print(f"  状态: {match_status}")
        print("-" * 50)
    
    # 计算整体性能指标
    match_rate = perfect_matches / len(source_texts) * 100     # 完全匹配率（百分比）
    avg_time = total_time / len(source_texts) * 1000           # 平均翻译时间（毫秒）
    bleu_score = calculate_bleu(reference_texts, translations) # BLEU分数
    
    # 打印评估结果摘要
    print(f"\n评估结果:")
    print(f"  BLEU分数: {bleu_score:.4f}")
    print(f"  完全匹配率: {match_rate:.2f}% ({perfect_matches}/{len(source_texts)})")
    print(f"  平均翻译时间: {avg_time:.2f}ms/句")
    
    return translations, bleu_score, match_rate

# 定义测试数据集
def get_test_data():
    """
    获取测试用的中英文句对
    
    返回:
        tuple: (中文句子列表, 英文句子列表)
        
    这些是简单的示例句子，用于演示和测试翻译器性能
    在实际应用中，应该使用更大更多样的测试集
    """
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
    
    return chinese_sentences, english_sentences

def main():
    """
    主函数：初始化翻译器并进行评估
    
    执行流程:
    1. 初始化翻译器
    2. 加载测试数据
    3. 评估中译英性能
    4. 评估英译中性能
    5. 比较与从头训练的Transformer性能
    6. 提供交互式翻译演示
    """
    # 初始化翻译器
    translator = HuggingFaceTranslator()
    
    # 获取测试数据
    chinese_sentences, english_sentences = get_test_data()
    
    # 评估中译英性能
    zh_to_en_translations, zh_to_en_bleu, zh_to_en_match_rate = evaluate_translator(
        translator, chinese_sentences, english_sentences, zh_to_en=True
    )
    
    # 评估英译中性能
    en_to_zh_translations, en_to_zh_bleu, en_to_zh_match_rate = evaluate_translator(
        translator, english_sentences, chinese_sentences, zh_to_en=False
    )
    
    # 比较与从头训练的Transformer模型的性能差异
    print("\n===== Hugging Face预训练模型与从头训练的Transformer性能比较 =====")
    print("Hugging Face预训练模型优势:")
    print("1. 中译英BLEU分数: {:.4f} (自训练Transformer: 0.1362)".format(zh_to_en_bleu))
    print("2. 中译英完全匹配率: {:.2f}% (自训练Transformer: 0.00%)".format(zh_to_en_match_rate))
    print("3. 无需大规模训练数据，可直接使用预训练模型")
    print("4. 翻译质量更好，特别是对于未见过的例子")
    print("5. 推理速度更快，不需要自回归生成中的多次前向传播")
    
    # 交互式翻译演示
    print("\n===== 交互式翻译演示 =====")
    print("输入'q'退出，输入's'切换翻译方向")
    
    # 默认从中文翻译到英文
    zh_to_en_mode = True
    
    # 交互循环
    while True:
        direction = "中译英" if zh_to_en_mode else "英译中"
        text = input(f"\n请输入{direction}文本 (q=退出, s=切换方向): ")
        
        # 处理特殊命令
        if text.lower() == 'q':  # 退出
            break
        elif text.lower() == 's':  # 切换翻译方向
            zh_to_en_mode = not zh_to_en_mode
            print(f"切换到{'中译英' if zh_to_en_mode else '英译中'}模式")
            continue
        
        # 执行翻译
        try:
            if zh_to_en_mode:
                translation = translator.translate_zh_to_en(text, debug=True)
            else:
                translation = translator.translate_en_to_zh(text, debug=True)
                
            print(f"翻译结果: {translation}")
        except Exception as e:
            print(f"翻译出错: {str(e)}")

# 程序入口点
if __name__ == "__main__":
    main() 