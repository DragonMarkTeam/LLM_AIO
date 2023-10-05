import argparse
import math
import os
def train_config():
    parser = argparse.ArgumentParser('Argument for training')
    # Setting training parameters
    parser.add_argument('--model', type=str, default="RNN",
                        help='Model need to train')
    parser.add_argument('--pretrain_tokenizer', type=str, default=None,
                        help='Pretrain tokenizer')
    parser.add_argument('--pretrain_model', type=str, default=None,
                        help='Pretrain model')
    
    parser.add_argument('--train_dataset', type=str, default=None,
                        help='Chooose dataset for training')
    parser.add_argument('--evaluation_dataset', type=str, default=None,
                        help='Chooose dataset for evaluation')
    
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=3,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--path', type=str, default='/data/data',
                        help='Path of folder')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to pre-trained model')
    
    

    opt = parser.parse_args()
    opt.save_folder = os.path.join(opt.path, "save_folder")
    opt.dataset_folder = os.path.join(opt.path, "dataset")
    opt.data_folder = os.path.join(opt.path, "preprocessing_data")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt