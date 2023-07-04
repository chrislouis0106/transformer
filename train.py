import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import configs, plot_loss
from datasets import TranslateDataset
from models import Transformer



def read_data(source_file, target_file):
    source_data = open(source_file,encoding='utf-8').read().strip().split("\n")
    target_data = open(target_file,encoding='utf-8').read().strip().split("\n")
    # print(source_data[1],len(source_data),len(target_data),target_data[0])
    # os._exit(0) 读取每个句子；
    return source_data, target_data


def validate_epoch(model, valid_loader, epoch, n_epochs, source_pad_id, target_pad_id, device):
    model.eval()
    total_loss = []
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validating epoch {epoch+1}/{n_epochs}")
    for i, batch in bar:
        source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
        target_input = target[:, :-1]
        source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
        preds = model(source, target_input, source_mask, target_mask)
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])

    valid_loss = sum(total_loss) / len(total_loss)
    return valid_loss, total_loss

# 每个epoch的计算过程；
def train_epoch(model, train_loader, optim, epoch, n_epochs, source_pad_id, target_pad_id, device):
    # 初始训练 train()指示器； 损失函数列表统计； 进度条与描述信息以及枚举训练加载器
    model.train()
    total_loss = []
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}/{n_epochs}")
    for i, batch in bar:
        # 遍历 bar 的 内容与 index； 把数据转移到device；
        source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
        # 从target输入中移除了最后一个token；常用于训练生成任务；使得模型学习生成序列基于输入文本；
        target_input = target[:, :-1]
        # 对source 和 target mask data ；得到的掩码数据为 与pad_id 不相等的地方赋值为1； target的处理也是一样；也就是将真实的数据标地位1
        source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
        # 数据输入model
        preds = model(source, target_input, source_mask, target_mask)
        # 梯度清零；
        optim.zero_grad()
        # from第二个token直到最后一个token；view(-1)结合所有的token到一个序列
        gold = target[:, 1:].contiguous().view(-1)
        # 计算预测与损失的loss，cross_entropy 的 ignore_index 参数；
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
        loss.backward()
        optim.step()
        total_loss.append(loss.item())
        # 检索最近的损失值从，
        bar.set_postfix(loss=total_loss[-1])
    
    train_loss = sum(total_loss) / len(total_loss)
    return train_loss, total_loss


'''
    train(model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        n_epochs=configs["n_epochs"],
        source_pad_id=source_tokenizer.pad_token_id,
        target_pad_id=target_tokenizer.pad_token_id,
        device=device,
        model_path=configs["model_path"],
        early_stopping=configs["early_stopping"]
    )
'''
def train(model, train_loader, valid_loader, optim, n_epochs, source_pad_id, target_pad_id, device, model_path, early_stopping):
    # 设置日志文件
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 设置初始值；
    best_val_loss = np.Inf
    best_epoch = 1
    count_early_stop = 0
    # 设置日志文件内容
    log = {"train_loss": [], "valid_loss": [], "train_batch_loss": [], "valid_batch_loss": []}
    for epoch in range(n_epochs):
        # 每个epoch 训练 and  验证；
        train_loss, train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optim=optim,
            epoch=epoch,
            n_epochs=n_epochs,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
            device=device
        )
        valid_loss, valid_losses = validate_epoch(
            model=model,
            valid_loader=valid_loader,
            epoch=epoch,
            n_epochs=n_epochs,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
            device=device
        )
        # 根据验证精度保存最好的摸i选哪个
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_epoch = epoch + 1
            # save model
            torch.save(model.state_dict(), model_path)
            print("---- Detect improment and save the best model ----")
            count_early_stop = 0
        else:
            # 达到early stop 则停止 训练；
            count_early_stop += 1
            if count_early_stop >= early_stopping:
                print("---- Early stopping ----")
                break
        # 每个epoch后都清楚缓存；会将已分配但未使用的GPU显存释放，并将其标记为空闲状态，以便其他操作可以使用
        torch.cuda.empty_cache()

        # 记录到日志文件并打印；
        log["train_loss"].append(train_loss)
        log["valid_loss"].append(valid_loss)
        log["train_batch_loss"].extend(train_losses)
        log["valid_batch_loss"].extend(valid_losses)
        log["best_epoch"] = best_epoch
        log["best_val_loss"] = best_val_loss
        log["last_epoch"] = epoch + 1

        with open(os.path.join(log_dir, "log.json"), "w") as f:
            json.dump(log, f)

        print(f"---- Epoch {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f} | Best Valid loss: {best_val_loss:.4f} | Best epoch: {best_epoch}")
    
    return log


def main():
    # read train and dev
    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])
    # vinai/phobert-base 是一个基于RoBERTa模型的预训练语言模型
    # bert的预训练语言模型；
    # 加载两种形式的token；
    '''
    BertTokenizerFast 是一个快速的BERT tokenizer，用于将文本序列转换为BERT模型所需的输入编码。
    它是Hugging Face Transformers库中的一部分。vocab_size=30522：指定词汇表的大小。BERT模型的默认词汇表大小为30522。
    '''
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    # vinai/phobert-base
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])

    # 使用自己定义的transformer模块；
    model = Transformer(
        # 可以source_tokenizer.vocab_size, 这种方式获取词汇表大小；
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,

        embedding_dim=configs["embedding_dim"],
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        num_layers=configs["n_layers"],
        num_heads=configs["n_heads"],
        dropout=configs["dropout"]
    )

    # 循环模型参数；使用Xavier_uniform 初始化；
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # "lr":0.0001,
    # beta ： 估计一阶矩（均值）和二阶矩（方差）的衰减率
    '''
    第一个参数 betas[0] 是用于计算一阶矩估计的衰减率，它决定了历史梯度的影响程度。较高的值会使优化器更加关注最近的梯度信息，而较低的值则会平均考虑更多的历史梯度信息。
    参数 betas[1] 是用于计算二阶矩估计的衰减率，它决定了历史梯度平方的影响程度。与一阶矩估计类似，较高的值会使优化器更加关注最近的梯度平方信息，而较低的值则会平均考虑更多的历史梯度平方信息。
    通过调整 betas 参数，您可以控制一阶矩估计和二阶矩估计的权重，以适应不同的优化任务和数据特征
    '''
    optim = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), eps=1e-9)

    # 通过序列的方式加载数据集；训练集
    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=train_src_data, 
        target_data=train_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
    )
    # 通过序列的方式加载验证集；
    valid_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=valid_src_data, 
        target_data=valid_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
    )
    # 根据配置文件设置train_loader 加载器； 传入数据集，batch_size  shuffle 等
    device = torch.device(configs["device"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    # 对于验证集也是一样，不同的是shuffle 为false；
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=configs["batch_size"],
        shuffle=False
    )
    # 模型to cuda config 是一个字典；
    model.to(configs["device"])
    # 调用train 函数；
    train(model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        n_epochs=configs["n_epochs"],
        source_pad_id=source_tokenizer.pad_token_id,
        target_pad_id=target_tokenizer.pad_token_id,
        device=device,
        model_path=configs["model_path"],
        early_stopping=configs["early_stopping"]
    )

    plot_loss(log_path="./logs/log.json", log_dir="./logs")

if __name__ == "__main__":
    main()
