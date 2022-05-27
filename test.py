# from transformers import AutoTokenizer
#
# # 加载预训练的tokenizer
# # 基于WordPiece Tokenization方法进行处理的
# # 加载了一个预训练的BERT模型，并创建了一个与该模型相对应的tokenizer对象。
# # tokenizer对象将输入的文本分割成一系列词片段（subwords）。每个词片段都以一个特殊的标记开头（[CLS]）和结束（[SEP]）。
# # Vocabulary映射：tokenizer对象使用预训练的BERT模型的词汇表（vocabulary）将每个词片段映射为一个唯一的整数ID。
# # tokenizer对象将最终的token序列和对应的ID返回给用户。
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#
# # 使用tokenizer对文本进行编码
# text = "Hello, how are you?"
# encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
#
# # 输出编码结果; 上面的句子被分解为了8个Token 基于 基于WordPiece的Bert的模型；
# print(encoded_input)
# # {'input_ids': tensor([[ 101, 7592, 1010, 2129, 2024, 2017, 1029,  102]]), # "Hello, how are you?"被分割成了几个词片段，然后每个词片段被映射为一个整数ID
# # 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),  #  这个键对应的tensor表示每个token属于哪个句子（在BERT模型中，句子对任务中会有两个句子，分别用0和1表示）。在这个例子中，只有一个句子，因此所有的token都被标记为0。
# # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])} # 这个键对应的tensor用于指示哪些位置是有效的（值为1），哪些位置是填充的（值为0）。在这个例子中，句子长度是8个token，所以所有位置都是有效的。

from transformers import AutoModel, AutoTokenizer
# 确保已经安装了Hugging Face的transformers库，它提供了对BERT和PhoBERT等预训练模型的支持。
# 这些预训练模型都在 hugging face 的transformer库中；
# AutoModel 加载预训练模型；
# AutoTokenizer 加载相应模型的分词器：包括词汇表等；
''' 
RoBERTa模型 embedding 层的定义与结构；word_embeddings: 这是一个词嵌入层，用于将输入的单词索引映射为对应的词向量。它是一个大小为(64001, 768)的嵌入矩阵，其中64001是词汇表的大小，768是每个词向量的维度。这个词嵌入层会将输入的单词索引转换为对应的词向量表示。
position_embeddings: 这是一个位置嵌入层，用于表示单词在输入句子中的位置信息。它是一个大小为(258, 768)的嵌入矩阵，其中258表示句子的最大长度（包括特殊标记），768是每个位置嵌入的维度。这个位置嵌入层会为每个位置生成一个对应的位置嵌入向量。
这是一个标记类型嵌入层，用于区分输入句子中不同类型的标记。在RoBERTa中，只使用了单个标记类型（通常用于区分句子对任务），因此它是一个大小为(1, 768)的嵌入矩阵。
LayerNorm: 这是一个层归一化层，用于归一化每个嵌入向量的维度，并通过可学习的参数进行缩放和平移
'''
model_name = "vinai/phobert-base" #
# 加载一个基于transformer的模型；包括模型的架构等；
model = AutoModel.from_pretrained(model_name)
# 得到model  token的说明 PhobertTokenizer(name_or_path='vinai/phobert-base', vocab_size=64000, model_max_length=256, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 使用加载的分词器对文本进行编码：text这个变量；
# 将输入文本编码为长度为128的序列，并进行填充和截断处理。
# encoder的输出结果为 list tokenid
text = "这是要进行编码的文本。"
# encoded_input = tokenizer.encode(text, padding=True, truncation=True, max_length=128)# 是长度为13的token id
# tokenizer 是一个PhobertTokenizer 类； 可调用对象，可以不直接初始化直接调用；创建了__call__ 方法；或者使用了@callable 装饰器；
encoded_input = tokenizer(text, paddding=True, truncation= True, max_length=128)
print(encoded_input)
import torch
# 把token id 转化成tensor
input_ids = torch.tensor([encoded_input['input_ids']])
outputs = model(input_ids)

# last_hidden_state  pooler_output  hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None 包括
print(outputs)
print(outputs.last_hidden_state.shape) # torch.Size([1, 13, 768])
print(outputs.pooler_output, outputs.pooler_output.shape) # torch.Size([1, 768])

input_ids = torch.tensor(encoded_input['input_ids']).unsqueeze(0)  # 添加一个维度作为批处理维度

with torch.no_grad():
    embeddings = model.embeddings(input_ids)

token_embeddings = embeddings.squeeze(0)  # 移除批处理维度
print(token_embeddings.shape)

'''
直接使用预训练模型的embedding到下游任务是一种常见的做法，并且通常可以取得不错的效果。
预训练模型的embedding已经通过大规模的无监督学习从大量的文本数据中学习到了丰富的语义信息，这些embedding可以作为输入数据的高级特征表示。
'''