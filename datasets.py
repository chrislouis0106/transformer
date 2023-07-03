import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re

class TranslateDataset(Dataset):
    # 初始化 加载数据的各个参数；继承Dataset;  包括源数据目标数据，源token 目标token 源序列长度， 目标序列长度
    def __init__(self, source_tokenizer, target_tokenizer, source_data=None, target_data=None, source_max_seq_len=256, target_max_seq_len=256, phase="train"):
        self.source_data = source_data
        self.target_data = target_data
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.phase = phase

    # 对于其他类型的任务，其文本处理是否需要进一步加强；
    def preprocess_seq(self, seq):
        # sub 替换： 使用正则表达式替换文本中的特殊字符，将它们替换为空格。
        # 具体的特征字符包括：星号、引号、换行符、反斜杠、省略号、加号、减号、斜杠、等号、括号、单引号、冒号、方括号、竖线、感叹号和分号
        seq = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq) # re.sub(r"[ ]+", " ", seq): 使用正则表达式将连续多个空格替换为单个空格
        seq = re.sub(r"\!+", "!", seq) #使用正则表达式将连续多个感叹号替换为单个感叹号
        seq = re.sub(r"\,+", ",", seq) # 使用正则表达式将连续多个问号替换为单个问号
        seq = re.sub(r"\?+", "?", seq) # 使用正则表达式将连续多个问号替换为单个问号
        seq = seq.lower()
        return seq

    # tokenize(text) 方法接受一个字符串参数 text，并将其拆分成一个个独立的词或子词
    # tokenizer 分词器调用 tokenize 方法 ； 并对分词进行切割；
    # 对分词后的列表 拼接 cls_token 与 stp_token 标记开头与结尾；
    def convert_line_uncased(self, tokenizer, text, max_seq_len):
        tokens = tokenizer.tokenize(text)[:max_seq_len-2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        # (max_seq_len - len(tokens))可能为0 否则则重复 pad_token max_seq_len- len(tokens) 次数；
        # 最后把padding的部分补充在token后面；
        tokens += [tokenizer.pad_token]*(max_seq_len - len(tokens))
        # 根据分词器将每个token映射为ID
        #  Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the vocabulary.
        token_idx = tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_idx

    # 根据dataset的构建方式：必须初始化三个函数 len init getitem; len 直接返回的是source data 的大小；
    def __len__(self):
        return len(self.source_data)

    # create decoder input mask 创建掩码矩阵；
    # 下三角矩阵常用语遮蔽后续位置的信息，防止信息泄露和未来信息的访问。
    # 使用.tril()方法，将张量转化为下三角矩阵形式，将上三角全部为0，下三角不变；
    def create_decoder_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len).tril()
        return mask
    
    def __getitem__(self, index):
        # source_seq, source_idx = self.convert_line_uncased(
        #     tokenizer=self.source_tokenizer, 
        #     text=self.preprocess_seq(self.source_data[index]), 
        #     max_seq_len=self.source_max_seq_len
        # )
        # target_seq, target_idx = self.convert_line_uncased(
        #     tokenizer=self.target_tokenizer, 
        #     text=self.preprocess_seq(self.target_data[index]), 
        #     max_seq_len=self.target_max_seq_len
        # )
        # 将index对应的source预处理移除无关词
        # 可以得到source text 的tensor id ；
        # padding的value 为 max_length 奇怪；
        source = self.source_tokenizer(
                text=self.preprocess_seq(self.source_data[index]),
                padding="max_length", 
                max_length=self.source_max_seq_len, 
                truncation=True, 
                return_tensors="pt"
            )
        
        if self.phase == "train":
            target = self.target_tokenizer(
                text=self.preprocess_seq(self.target_data[index]),
                padding="max_length",
                max_length=self.target_max_seq_len,
                truncation=True,
                return_tensors="pt" # return_tensors="pt"是指定了Tokenizer返回的编码结果的数据类型为PyTorch张量 pytorch tensor ；
            )
            # 原始数据 与 source  id tensor ；
            return {
                "source_seq": self.source_data[index],
                # Tokenizer的batch_encode_plus方法返回的，表示一个批次中的多个样本，因此它是一个二维张量。
                "source_ids": source["input_ids"][0],
                "target_seq": self.target_data[index],
                # 通过 [0] 索引可以获取到当前样本的编码结果。
                "target_ids": target["input_ids"][0],
                }
        else:
            # 只返回source
            return {
                "source_seq": self.source_data[index],
                "source_ids": source["input_ids"][0],
            }


def main():
    from utils import configs

    def read_data(source_file, target_file):
        source_data = open(source_file,encoding='utf-8').read().strip().split("\n")
        target_data = open(target_file,encoding='utf-8').read().strip().split("\n")
        return source_data, target_data

    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])

    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])
    # 数据包括原始数据，与原始数据的token id ；
    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=train_src_data, 
        target_data=train_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"]
    )
    # 可以通过索引查看；
    print(train_dataset[0])

    # 设置数据加载方式；batch size 与 shuffle ；
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    # 在训练阶段加载四个原始数据：source sourceID target targetID ；
    for batch in train_loader:
        print(batch["source_seq"])
        print(batch["target_seq"])
        print(batch["source_ids"])
        print(batch["target_ids"])
        break

    '''
    import ipdb; ipdb.set_trace() 是一种在代码中插入调试断点的方法。
    当代码执行到这一行时，它会触发一个调试器，例如 ipdb，允许您逐行执行代码并查看变量的值，以便进行调试和排查问题。
    在服务器上也可以通过这个方式查看中间变量！
    '''
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()