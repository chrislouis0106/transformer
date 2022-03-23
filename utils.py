import torch

configs = {
    "train_source_data":"./data_en_vi/train.en",
    "train_target_data":"./data_en_vi/train.vi",
    "valid_source_data":"./data_en_vi/tst2013.en",
    "valid_target_data":"./data_en_vi/tst2013.vi",
    "source_tokenizer":"bert-base-uncased",
    "target_tokenizer":"vinai/phobert-base",
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "batch_size":40,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.1,
    "lr":0.0001,
    "n_epochs":50,
    "print_freq": 5,
    "beam_size":5,
    "model_path":"./model_transformer_translate_en_vi.pt"
}
