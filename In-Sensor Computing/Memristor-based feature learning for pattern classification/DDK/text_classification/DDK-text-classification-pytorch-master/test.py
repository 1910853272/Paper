import os
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from build_vocab import build_dictionary
from dataset import CustomTextDataset, collate_fn
from model import RCNN
from trainer import train, evaluate
from utils import read_file
from thop import clever_format
from thop import profile
logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed_id):
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    #if args.n_gpu > 0:
    #    torch.cuda.manual_seed_all(args.seed)

def cal_flops(model):
    input = torch.randint(100, (1, 64)).to(args.device)
    flops, params = profile(model, inputs=(input,args))
    macs, params = clever_format([flops, params], "%.3f")
    a = 0
def main(args, seed_id):
    model = RCNN(vocab_size=args.vocab_size,
                 embedding_dim=args.embedding_dim,
                 hidden_size=args.hidden_size,
                 hidden_size_linear=args.hidden_size_linear,
                 class_num=args.class_num,
                 dropout=args.dropout,
                 DDK_type = args.DDK_type).to(args.device)
    cal_flops(model)
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model, dim=0)

    train_texts, train_labels = read_file(args.train_file_path)
    word2idx = build_dictionary(train_texts, vocab_size=args.vocab_size)
    logger.info('Dictionary Finished!')
    # Test
    if args.test_set:
        test_texts, test_labels = read_file(args.test_file_path)
        test_dataset = CustomTextDataset(test_texts, test_labels, word2idx)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=lambda x: collate_fn(x, args),
                                     batch_size=args.batch_size,
                                     shuffle=True)

        model.load_state_dict(torch.load(os.path.join(args.model_save_path, str(seed_id) + "_best.pt")))
        _, accuracy, precision, recall, f1, cm = evaluate(model, test_dataloader, args)
        logger.info('-' * 50)
        logger.info(
            f'|* TEST SET *| |ACC| {accuracy:>.4f} |PRECISION| {precision:>.4f} |RECALL| {recall:>.4f} |F1| {f1:>.4f}')
        logger.info('-' * 50)
        logger.info('---------------- CONFUSION MATRIX ----------------')
        for i in range(len(cm)):
            logger.info(cm[i])
        logger.info('--------------------------------------------------')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed', nargs='+', type=int)
    parser.add_argument('--test_set', action='store_true', default=True)

    # data
    parser.add_argument("--train_file_path", type=str, default="/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master/data/train.csv")
    parser.add_argument("--test_file_path", type=str, default="/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master/data/test.csv")
    parser.add_argument("--model_save_path", type=str, default="/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master/model_saved_word_dim")
    parser.add_argument("--num_val_data", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)

    # model
    parser.add_argument("--DDK_type", type=str, default='word_dim')
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--hidden_size_linear", type=int, default=512)
    parser.add_argument("--class_num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # training
    parser.add_argument("--epochs", type=int, default=20)
    #parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    for seed_id in args.seed:
        set_seed(seed_id)

        main(args, seed_id)