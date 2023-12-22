
TRAIN_SAMPLE_PATH = '/path'
DEV_SAMPLE_PATH= '/path'
TEST_SAMPLE_PATH = '/path'
LABEL_PATH = '/path'            #"NFR_class.txt"


BERT_PAD_ID = 0
TEXT_LEN = 512

BERT_MODEL = '/path'

MODEL_DIR1 = '/path'
# MODEL_DIR2 = '/path'

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 37
FILTER_SIZES = [2, 3, 4]
FILTER_SIZES2 = [2, 3, 4, 5]
NUM_HEADS = 2
DROPOUT=0.5
ATTENTION_HIDDEN_SIZE=64

EPOCH = 20
LR = 0.001

import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Corresponding numbered graphics cards

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.tensor([1,2,3]).to(DEVICE))