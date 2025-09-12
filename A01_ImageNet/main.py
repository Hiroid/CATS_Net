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
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parent.parent

def main(args):
    # Adjust args if using one-hot initialization
    if args.symbol_init_type == 'one_hot':
        args.symbol_size = args.num_classes  # Set to num_classes for Dataset
    
    str_time = gettime(time.localtime(time.time()))
    log_dir = project_root / "Results" / "log"
    model_dir = project_root / "Results" / "param"

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    fix_str = ""
    if args.fix_ts:
        fix_str += "_fixtsTrue"
    if args.fix_ts_ca:
        fix_str += "_fixtscaTrue"
    if args.fix_symbol_set:
        fix_str += "_fixsymbolTrue"

    base_file_name = f"{args.dataset}_ss{args.symbol_size}_fixfe_{args.model_name}_mlp{args.mlp_layers}_hidden{args.hidden_dim}{fix_str}_init{args.symbol_init_type}_{str_time}"
    model_file_name = f"{base_file_name}.pt"

    if args.exp_prefix is None:
        log_file_name = f"{base_file_name}.log"
    else:
        log_file_name = f"{args.exp_prefix}_{base_file_name}.log"
    logger = get_logger(os.path.join(project_root, "Results", "log", log_file_name))

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        raise RuntimeError
    
    ### print all args
    argument.print_args(args, logger)
    logger.info(f'Device is {device}')
    
    ### data preparation
    data_train, data_test = data.mkdataset(args)
    
    # WebDataset objects don't have len() method, so we handle this case
    try:
        train_len = len(data_train)
        test_len = len(data_test)
        logger.info(f'length of train is {train_len}, length of test is {test_len}')
    except TypeError:
        logger.info('Using WebDataset format - dataset length not available')

    net = model.cats_net(
        symbol_size = args.symbol_size, 
        mlp_layers = args.mlp_layers,
        hidden_dim = args.hidden_dim,
        num_classes = args.num_classes, 
        fix_fe = args.fix_fe, 
        fe_type = args.model_name,
        pretrain = args.use_pretrain,
    )
    
    # Initialize symbol_set based on the specified type
    init_type = args.symbol_init_type
    if args.custom_symbol_path is not None:
        init_type = 'custom'
    
    net.init_symbol_set(init_type=init_type, custom_path=args.custom_symbol_path)
    logger.info(f'Symbol set initialized with type: {init_type}')
    
    if args.load_model != None: net.load_state_dict(torch.load(args.load_model))
    loss = nn.__dict__[args.loss_type]()

    ### optimizer
    param_opti, symbol_opti = utils.get_optimizer(
        net, 
        args.optimizer_type, 
        args.learning_rate, 
        args.momentum, 
        args.weight_decay, 
        args.fix_ts
    )

    utils.train(
        net, 
        loss, 
        data_train, 
        data_test, 
        param_opti, 
        symbol_opti, 
        device, 
        args, 
        logger, 
        ckpt = os.path.join(project_root, "Results", "param", model_file_name)
    )
    
    torch.save(
        net.state_dict(), 
        os.path.join(project_root, "Results", "param", model_file_name)
    )

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
