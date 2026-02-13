import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage):

    input_argument = (
        '--input=%s '
        '--model_prefix=%s '
        '--vocab_size=%s '
        '--model_type=%s '
        '--character_coverage=%s '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    )

    # 将传入参数填充到命令字符串
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)

    # 开始训练；会在当前工作目录下生成 <model_name>.model / <model_name>.vocab
    spm.SentencePieceTrainer.Train(cmd)


def run():
    # 英文分词器
    en_input = '../data/corpus.en'
    en_vocab_size = 32000
    en_model_name = 'eng'
    en_model_type = 'bpe'
    en_character_coverage = 1.0

    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    # 中文分词器配置
    ch_input = '../data/corpus.ch'
    ch_vocab_size = 32000
    ch_model_name = 'chn'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995

    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)

def test():
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今日抵达夏威夷。"
    sp.Load("./chn.model")
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))
    a = [12907, 277, 7419, 7318, 18384, 28724]
    print(sp.DecodeIds(a))

def chinese_tokenizer_load():
    """
    加载中文分词器模型
    该函数用于加载预训练的中文SentencePiece分词器模型，用于文本预处理和分词。
    返回:
        sp_chn: 加载好的SentencePieceProcessor对象，可用于中文文本的分词处理
    使用方法:
        tokenizer = chinese_tokenizer_load()
        tokens = tokenizer.tokenize("中文文本")
    """

    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("./tokenizer/chn"))
    return sp_chn


def english_tokenizer_load():
    """
    加载英文分词器模型
    该函数用于加载英文的SentencePiece分词器模型，该模型用于将英文文本转换为token序列。
    返回:
        SentencePieceProcessor: 加载了英文模型的SentencePieceProcessor对象，可用于英文文本的分词处理
    示例:
        tokenizer = english_tokenizer_load()
        tokens = tokenizer.encode("Hello world")
    """
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('{}.model'.format("./tokenizer/eng"))
    return sp_eng

if __name__ == "__main__":
    run()
