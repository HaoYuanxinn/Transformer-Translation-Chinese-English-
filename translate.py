import torch
import config
import logging
import numpy as np
from tokenizer.tokenize import english_tokenizer_load
from transformer_model import make_model
from tokenizer.tokenize import chinese_tokenizer_load
from beam_decoder import beam_search

logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s-%(funcName)s:%(lineno)d', level=logging.INFO)


def translate(src, model):
    """用训练好的模型进行预测单句，打印模型翻译结果"""

    sp_chn = chinese_tokenizer_load()

    with torch.no_grad():
        model.load_state_dict(torch.load(config.test_model_path))
        model.eval()

        src_mask = (src != 0).unsqueeze(-2)

        decode_result, _ = beam_search(
            model,
            src,
            src_mask,
            config.max_len,  # 最大翻译长度
            config.padding_idx,  # 填充符号的索引
            config.bos_idx,  # 句子开始符号的索引
            config.eos_idx,  # 句子结束符号的索引
            config.beam_size,  # 束搜索的大小
            config.device  # 设备（CPU或GPU）
        )

        decode_result = [h[0] for h in decode_result]
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
        return translation[0]

def one_sentence_translate(sent):
    """翻译单句英文"""

    model = make_model(
        config.src_vocab_size,  # 源语言词汇表大小
        config.tgt_vocab_size,  # 目标语言词汇表大小
        config.n_layers,  # 模型的层数
        config.d_model,  # 模型的维度（通常是隐藏层的大小）
        config.d_ff,  # 前馈网络的维度
        config.n_heads,  # 注意力头的数量
        config.dropout  # dropout比率
    )

    BOS = english_tokenizer_load().bos_id()
    EOS = english_tokenizer_load().eos_id()
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    return translate(batch_input, model)

def translate_example():
    """单句翻译示例"""

    while True:
        sent = input("请输入英文句子进行翻译：")
        translation = one_sentence_translate(sent)
        print("翻译结果：", translation)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    translate_example()