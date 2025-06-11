import torch
from torch import nn
from . import utils
from . import model
from . import data
from . import argument
import datetime
import time
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    str_time = gettime(time.localtime(time.time()))
    if args.exp_prefix is None:
        logger = get_logger(f'../Results/log/{args.dataset}_ss{args.symbol_size}_fixfe_{str_time}.log')
    else:
        logger = get_logger(f'../Results/log/{args.exp_prefix}_{args.dataset}_ss{args.symbol_size}_fixfe_{str_time}.log')
    
    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        raise RuntimeError
    
    ### print all args
    argument.print_args(args, logger)
    logger.info(f'Device is {device}')
    
    ### data preparation
    data_train, data_test = data.mkdataset(args)
    logger.info(f'length of train is {(len(data_train))}, length of test is {(len(data_test))}')

    ### build network
    if args.bidir:
        net = model.bidir_sea_net(
            symbol_size = args.symbol_size, 
            num_classes = args.num_classes, 
            fix_fe = args.fix_fe, 
            fe_type = args.model_name,
            pretrain = args.use_pretrain,
        )
    else:
        net = model.sea_net(
            symbol_size = args.symbol_size, 
            num_classes = args.num_classes, 
            fix_fe = args.fix_fe, 
            fe_type = args.model_name,
            pretrain = args.use_pretrain,
        )
    if args.load_model != None: net.load_state_dict(torch.load(args.load_model))
    loss = nn.__dict__[args.loss_type]()

    ### optimizer
    param_opti, symbol_opti = utils.get_optimizer(net, args.optimizer_type, args.learning_rate, args.momentum, args.weight_decay, args.random_ts)

    utils.train(net, loss, data_train, data_test, param_opti, symbol_opti, device, args, logger, ckpt = f"../Results/param/{args.dataset}_ss{args.symbol_size}_fixfe_{str_time}.pt")
    # utils.train_concept_first(net, loss, data_train, data_test, param_opti, symbol_opti, device, args, logger, ckpt = f"../Results/param/{args.dataset}_ss{args.symbol_size}_fixfe_{str_time}.pt")
    
    torch.save(net.state_dict(), f"../Results/param/{args.dataset}_ss{args.symbol_size}_fixfe_{str_time}.pt")

    logger.info(f'Training done!') 
    

def prt_time():
    print('='*100)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('='*100)

def gettime(time):
    timestr = str('%04d'%time.tm_year) +  str('%02d'%time.tm_mon) + str('%02d'%time.tm_mday) + str('%02d'%time.tm_hour) + str('%02d'%time.tm_min) + str('%02d'%time.tm_sec)
    return timestr

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

if __name__ == '__main__':
    
    prt_time()
    
    args = argument.parser()
    main(args)

    prt_time()
