import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizer.tokenize import english_tokenizer_load
from tokenizer.tokenize import chinese_tokenizer_load
import config
DEVICE = config.device


def subsequent_mask(size):
    """
    该函数生成一个遮蔽矩阵，防止解码时看到未来的词（自回归）。
    目的是为了在训练解码器时，确保每个位置的预测只能基于当前及之前的词。
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """
    Class for holding a batch of data with corresponding masks during training.
    该类用于存储一个训练批次的数据，包括源语言和目标语言的文本、token以及mask。
    """
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        """
            初始化Batch类，生成对应的源语言和目标语言数据
            :param src_text: 源语言（英语）的文本
            :param trg_text: 目标语言（中文）的文本
            :param src: 源语言的输入数据（tensor格式）
            :param trg: 目标语言的输入数据（tensor格式），如果有
            :param pad: padding值（用于填充句子时的标记）
        """
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if trg is not None:
            trg = trg.to(DEVICE)
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建目标语言的mask，屏蔽padding和未来的词
        :param tgt: 目标语言的token序列
        :param pad: padding标记
        :return: 目标语言的mask
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)

        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    """
        自定义数据集类，用于加载机器翻译任务中的数据。该类继承自PyTorch的Dataset类。
        主要功能包括：加载英文和中文句子，使用分词器进行分词，将句子转换为token ID，并填充句子长度。
    """
    def __init__(self, data_path):
        """
            初始化函数，加载数据集和分词器
            :param data_path: 数据集文件路径（json格式）
        """
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()
        self.BOS = self.sp_eng.bos_id()
        self.EOS = self.sp_eng.eos_id()

    @staticmethod
    def len_argsort(seq):
        """
        对句子按照长度进行排序，并返回排序后句子的索引位置。
        :param seq: 需要排序的句子（列表形式）
        :return: 排序后的索引列表
        """
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """
        获取数据集，并根据英文句子长度对数据进行排序。
        :param data_path: 数据集的路径
        :param sort: 是否对数据按英文句子长度进行排序
        :return: 英文句子列表和中文句子列表
        """
        dataset = json.load(open(data_path, 'r',encoding="utf-8"))
        out_en_sent = []
        out_cn_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_cn_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        """
        获取指定索引的英文和中文句子。
        :param idx: 数据索引
        :return: 英文和中文句子对
        """
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        """
            返回数据集的大小（句子的数量）
        """
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        """
            定义如何将数据样本合并成一个batch，进行填充、编码等操作。
            :param batch: 一个batch的样本
            :return: 返回处理后的Batch对象
        """
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)