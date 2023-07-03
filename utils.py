import torch
import json
import os

# 以键值对的形式保存相应的配置参数；
configs = {
    "train_source_data":"./data_en_vi/train.en",
    "train_target_data":"./data_en_vi/train.vi",
    "valid_source_data":"./data_en_vi/tst2013.en",
    "valid_target_data":"./data_en_vi/tst2013.vi",
    "source_tokenizer":"bert-base-uncased",
    # vinai/phobert-base 是一个基于RoBERTa模型的预训练语言模型
    "target_tokenizer":"vinai/phobert-base",
    # 最大Token的长度；输入 and 输出；
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "batch_size":1,
    # "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "device": "cpu",
    "embedding_dim": 512,
    # transformer的层数
    "n_layers": 6,
    # multi-head 的个数；
    "n_heads": 8,
    "dropout": 0.1,
    "lr":0.0001,
    "n_epochs":50,
    "print_freq": 5,
    "beam_size":3,
    "model_path":"./model_transformer_translate_en_vi.pt",
    "early_stopping":5
}

import matplotlib.pyplot as plt

# visualize log
# 只是使用plot 包 绘制 损失函数；并保存在日志路径中；
def plot_loss(log_path, log_dir):
    log = json.load(open(log_path, "r"))

    # 打开画布；
    plt.figure()
    # 画布的两个数据 x 和 y 轴
    plt.plot(log["train_loss"], label="train loss")
    plt.plot(log["valid_loss"], label="valid loss")
    # 标题，标签等
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # 图例；
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_epoch.png"))

    # plot batch loss
    plt.figure()
    lst = log["train_batch_loss"]
    # x 轴； 像是某个batch的loss；
    n = int(len(log["train_batch_loss"]) / len(log["valid_batch_loss"]))
    train_batch_loss = [lst[i:i + n][0] for i in range(0, len(lst), n)]
    # 绘制图的相关信息；
    plt.plot(train_batch_loss, label="train loss")
    plt.plot(log["valid_batch_loss"], label="valid loss")
    plt.title("Loss per batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_batch.png"))