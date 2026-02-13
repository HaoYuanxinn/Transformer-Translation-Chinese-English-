import json

if __name__ == "__main__":
    files = ['train', 'dev', 'test']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines = []
    en_lines = []

    for file in files:
        corpus = json.load(open('./json/' + file + '.json', 'r', encoding="utf-8"))
        for item in corpus:
            en_lines.append(item[0] + '\n')
            ch_lines.append(item[1] + '\n')

    # 将中文句子写入文件
    with open(ch_path, "w", encoding="utf-8") as fch:
        fch.writelines(ch_lines)

    # 将英文句子写入文件
    with open(en_path, "w", encoding="utf-8") as fen:
        fen.writelines(en_lines)

    print("lines of Chinese: ", len(ch_lines))
    print("lines of English: ", len(en_lines))

