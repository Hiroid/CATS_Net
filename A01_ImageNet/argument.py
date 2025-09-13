import argparse

def parser():
    parser = argparse.ArgumentParser(description = 'CATS-Net on ImageNet1k concept abstraction task')
    parser.add_argument('--dataset', type = str, default = 'imagenet1k', choices = ['cifar10', 'cifar100', 'imagenet1k', 'imagenet100'], help = 'Which dataset to be used')
    parser.add_argument('-data_root', type = str, default = '/data', help = 'the directory to save the dataset')

    # parameters for training 
    parser.add_argument('--model_name', '-mn', type = str, default = 'resnet50', choices = ['resnet50', 'resnet18', 'vit_b_16'])
    parser.add_argument('--num_epochs', '-ne', type = int, default = 5, help = 'the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type = float, default = 0.001,  help = 'learning rate')
    parser.add_argument('--batch_size', '-bs', type = int, default = 512, help = 'batch size')
    parser.add_argument('--optimizer_type', '-ot', type = str, default = 'Adam', choices = ['SGD', 'Adam'], help = 'the type of optimizer')
    parser.add_argument('--momentum', '-m', type = float, default = 0.9, help = 'the momentum parameter')
    parser.add_argument('--print_frequency', '-pf', type = int, default = 1, help = 'the print frequency of training stage')
    parser.add_argument('--weight_decay', '-dw', type = float, default = 1e-4, help = 'the Regularization value')
    parser.add_argument('--load_model', '-lm', type = str, default = None, help = 'whether load checkpoint')
    parser.add_argument('--symbol_size', '-ss', type = int, default = 20, help = 'symbol size of CATS-net')
    parser.add_argument('--mlp_layers', '-ml', type = int, default = 3, help = 'num of mlp layers')
    parser.add_argument('--hidden_dim', '-hd', type = int, default = 100, help = 'hidden dimension of mlp layers')
    parser.add_argument('--num_classes', '-nc', type = int, default = 1000, help = 'num of symbols')
    parser.add_argument('-p', type = float, default = 0.5, help = 'the possibility for sampling negative symbol')
    parser.add_argument('--use_orthg', '-uc', default = False, action = 'store_true')
    parser.add_argument('--cntrst', type = int, default = None, help = 'use how many negative sample on each sample')
    parser.add_argument('--worst_neg', '-wn', default = False, action = 'store_true')
    parser.add_argument('--metric', type = str, default = 'euclidean', choices = ['euclidean', 'cosine'])
    parser.add_argument('--fix_fe', '-ff', default = False, action = 'store_true')
    parser.add_argument('--loss_type', type = str, default = 'CrossEntropyLoss', choices = ['CrossEntropyLoss', 'MSELoss'])
    parser.add_argument('--use_pretrain', '-up', default = False, action = 'store_true')
    parser.add_argument('--bidir', '-bi', default = False, action = 'store_true', help = 'use the concept flow from the other direction')
    parser.add_argument('--fix_ts', '-rt', default = False, action = 'store_true')
    parser.add_argument('--fix_ts_ca', '-rtc', default = False, action = 'store_true')
    parser.add_argument('--exp_prefix', '-ep', type = str, default = None, help = 'prefix of the exp')
    parser.add_argument('--fix_symbol_set', '-fss', default = False, action = 'store_true', help = 'whether to fix symbol_set during training')
    parser.add_argument('--symbol_init_type', '-sit', type = str, default = 'random', choices = ['random', 'one_hot', 'word2vec'], help = 'symbol_set initialization type')
    parser.add_argument('--custom_symbol_path', '-csp', type = str, default = None, help = 'path to custom symbol_set file')
    parser.add_argument('--joint_training', '-jt', default = False, action = 'store_true', help = 'whether to train symbol_set and network parameters jointly')
    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
