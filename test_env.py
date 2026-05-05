import torch
import numpy as np
import tqdm
import sentencepiece
import sacrebleu
import matplotlib

print("===== Environment Check =====")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))

print("numpy:", np.__version__)
print("tqdm:", tqdm.__version__)
print("sentencepiece:", sentencepiece.__version__)
print("sacrebleu:", sacrebleu.__version__)
print("matplotlib:", matplotlib.__version__)