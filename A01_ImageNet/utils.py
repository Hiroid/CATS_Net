from ast import Raise
import torch
import random
import time
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import copy
from tqdm import tqdm

def get_optimizer(net, optimizer_type, lr, momentum, wd, random_ts = False):
    if random_ts:
        param_list = [param for name, param in net.named_parameters() if 'cdp_' in name]
    else:
        param_list = [param for name, param in net.named_parameters() if 'symbol_set' not in name]

    param_opti = torch.optim.__dict__[optimizer_type](
        param_list, 
        lr = lr, 
        weight_decay = wd
    )
    symbol_opti = torch.optim.__dict__[optimizer_type](
        [net.symbol_set], 
        lr = lr, 
        weight_decay = wd
    )
    
    return param_opti, symbol_opti

def get_indices(dataset, class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_name:
            indices.append(i)
    return indices

def calc_dis(symbol_set, metric):
    vectors = symbol_set.cpu().detach().numpy()
    dis_matrix = pairwise_distances(vectors, metric = metric)

    return dis_matrix

def get_index(x, num_all, p_base, p_i, dis_matrix = None):
    if p_i >= p_base: 
        return x
    
    if dis_matrix is not None:
        symbol_x_dis = copy.deepcopy(dis_matrix[x, :])
        # euclidean
        symbol_x_dis[x] = float('inf')
        return symbol_x_dis.argmin()
        # # cosine
        # symbol_x_dis[x] = float('-inf')
        # return symbol_x_dis.argmax()

    while True:
        index = random.randint(0, num_all - 1)
        if x != index: return index

def get_batch(y, symbol_set, negative = False, p = 0.5, cntrst = None, worst_neg = False, metric = 'cosine'):
    batch_size = len(y)
    num_symbols = symbol_set.shape[0] # 10
    dis_matrix = calc_dis(symbol_set, metric) if worst_neg else None
    if cntrst == None:
        if negative:
            # p_base = torch.rand(1) 
            p_base = torch.tensor([p])
        else:
            p_base = torch.tensor([-1])
        p_batch = torch.rand(batch_size) # range from 0 to 1

        symbol_batch = torch.cat([symbol_set[get_index(int(y[i]), num_symbols, p_base, p_batch[i], dis_matrix), :].unsqueeze(0) for i in range(batch_size)], dim = 0)
        y_binary = torch.cat([torch.tensor([float(p_batch[i] < p_base), float(p_batch[i] >= p_base)]).unsqueeze(0) for i in range(batch_size)], dim = 0)

        return y_binary, symbol_batch
    else:
        # original sample, positive
        symbol_batch = torch.cat([symbol_set[int(y[i]), :].unsqueeze(0) for i in range(batch_size)], dim = 0)
        y_binary = torch.cat([torch.tensor([0.0, 1.0]).unsqueeze(0) for i in range(batch_size)], dim = 0)

        # negative samples
        for cnt in range(cntrst):
            symbol_tmp = torch.cat([symbol_set[get_index(int(y[i]), num_symbols, 1, 0, dis_matrix), :].unsqueeze(0) for i in range(batch_size)], dim = 0)
            y_tmp = torch.cat([torch.tensor([1.0, 0.0]).unsqueeze(0) for i in range(batch_size)], dim = 0)

            symbol_batch = torch.cat([symbol_batch, symbol_tmp], dim = 0)
            y_binary = torch.cat([y_binary, y_tmp], dim = 0)
        
        return y_binary, symbol_batch

def get_batch_fast(y, symbol_set, negative = False, p = 0.5):
    device = symbol_set.device
    batch_size = len(y)
    num_symbols = symbol_set.shape[0] # 10

    y_binary = torch.zeros(batch_size, 2).to(device)
    if negative:    
        y_binary[:int(batch_size*(1-p)), 1] = 1.0
        y_binary[int(batch_size*(1-p)):, 0] = 1.0
        y_diff = (y + torch.randint(1, num_symbols, (batch_size,)).to(device)) % num_symbols
        all_indices = torch.cat([y[:int(batch_size*(1-p))], y_diff[int(batch_size*(1-p)):]])
        symbol_batch = symbol_set[all_indices]
    else:
        y_binary[:, 1] = 1.0
        symbol_batch = symbol_set[y]
    return y_binary, symbol_batch

def evaluate_accuracy(eval_set, net, batch_size = 128, stat = False, use_feature = False, use_iter = False):
    device = list(net.parameters())[0].device
    num_symbols = net.symbol_set.shape[0] # 10
    if use_iter == True:
        data_iter = eval_set
    else:
        data_iter = torch.utils.data.DataLoader(eval_set, batch_size = batch_size, num_workers = 4, shuffle = False)
    statlist = [0 for i in range(num_symbols)]
    test_acc_sum, n = 0.0, 0
    with torch.no_grad():
        for tmp in tqdm(data_iter):
            X = tmp[0]
            y = tmp[1]
            net.eval() # shut down dropout
            X = X.to(device)
            y = y.to(device)
            # y_binary, symbol_batch = get_batch(y, net.symbol_set, negative = True, p = 0.5)
            y_binary, symbol_batch = get_batch_fast(y, net.symbol_set, negative = True, p = 0.5)
            y_binary = y_binary.to(device)
            symbol_batch = symbol_batch.to(device)
            
            # predict using y
            if use_feature == False:
                y_hat = net(X, symbol_batch)
            else:
                y_hat = net.feature_forward(X, symbol_batch)
            idx = (y_hat.argmax(dim = 1) == y_binary.argmax(dim = 1))
            
            test_acc_sum += idx.sum().cpu().item()
            # print(y)
            for i in range(len(y)):
                statlist[int(y[i])] += int(idx[i])
            net.train() # switch back to training mode
            n += y.shape[0]

    if stat == False: return test_acc_sum / n
    else: return test_acc_sum / n, statlist

def evaluate_accuracy_noy(eval_set, net, batch_size = 1, stat = False):
    device = list(net.parameters())[0].device
    num_symbols = net.symbol_set.shape[0] # 10
    data_iter = torch.utils.data.DataLoader(eval_set, batch_size = batch_size, num_workers = 4, shuffle = False)
    statlist = [0 for i in range(num_symbols)]
    test_acc_sum, n = 0.0, 0
    with torch.no_grad():
        for tmp in tqdm(data_iter):
            X = tmp[0] 
            y = tmp[1] 
            net.eval() # shut down dropout
            X = X.to(device)
            y = y.to(device)
            
            # predict w/o using y
            y_hat = net.cls(X)
            idx = (y_hat == y)

            test_acc_sum += idx.sum().cpu().item()
            # print(y)
            for i in range(len(y)):
                statlist[int(y[i])] += int(idx[i])
            net.train() # switch back to training mode
            n += y.shape[0]

    if stat == False: return test_acc_sum / n
    else: return test_acc_sum / n, statlist


def train(
    net, 
    loss, 
    train_set, 
    test_set, 
    param_opti, 
    symbol_opti, 
    device, 
    args, 
    logger = None, 
    ckpt = None,
):
    train_iter = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, num_workers = 8, shuffle = True)
    net.to(device)
    batch_ckpt_count = 0

    for epoch in range(args.num_epochs):
        if not args.random_ts_cdp:
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            cpu_gpu_time = 0.0
            make_batch_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            grad_step_time = 0.0
            other_stat_tims = 0.0
            for tmp in tqdm(train_iter):
                time1 = time.time()
                X_ = tmp[0] 
                y = tmp[1]
                net.train()
                X = X_.to(device)
                y = y.to(device)
                
                time2 = time.time()
                y_binary, symbol_batch = get_batch_fast(y, net.symbol_set, negative = True, p = args.p)
                
                time3 = time.time()
                y_hat = net(X, symbol_batch)
                l = loss(y_hat, y_binary)
                if args.use_orthg: l += net.symbol_orthg()
                
                time4 = time.time()
                param_opti.zero_grad()
                symbol_opti.zero_grad()
                l.backward()
                
                time5 = time.time()
                param_opti.step()

                time6 = time.time()
                train_l_sum += l.cpu().item()
                idx = (y_hat.argmax(dim = 1) == y_binary.argmax(dim = 1))
                train_acc_sum += idx.sum().cpu().item()
                n += y.shape[0] if args.cntrst is None else y.shape[0] * (args.cntrst + 1)
                batch_count += 1
                
                cpu_gpu_time += time2 - time1
                make_batch_time += time3 - time2
                forward_time += time4 - time3
                backward_time += time5 - time4
                grad_step_time += time6 - time5
                time7 = time.time()
                other_stat_tims += time7 - time6

                # torch.save(net.state_dict(), f"./param/sea-net_imagenet1k_ne5_trail5_ckpt{batch_ckpt_count}.pt")
                # batch_ckpt_count += 1

            if args.print_frequency != 0 and (epoch + 1) % args.print_frequency == 0:
                time8 = time.time()
                test_acc = evaluate_accuracy(test_set, net, batch_size = args.batch_size) if test_set != None else 0
                time9 = time.time()
                eval_time = time9 - time8
                if logger is None:
                    print('epoch %d, network training phase, loss %.4f, train acc %.4f, test acc %.4f, time %.2f sec'
                        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
                else:
                    logger.info(f'epoch {epoch + 1}, network phase, loss {(train_l_sum / batch_count):.4f}, train acc {(train_acc_sum / n):.4f}, test acc given symbol {test_acc:.4f}')
                    logger.info(f'total time {(time.time() - start):.3f}sec, IO {cpu_gpu_time:.3f}sec, make batch {make_batch_time:.3f}sec, forward {forward_time:.3f}sec, backward {backward_time:.3f}sec, grad step {grad_step_time:.3f}sec, others stat {other_stat_tims:.3f}sec, eval {eval_time:.3f}sec')
        
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        cpu_gpu_time = 0.0
        make_batch_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        grad_step_time = 0.0
        other_stat_tims = 0.0
        for tmp in tqdm(train_iter):
            time1 = time.time()
            X_ = tmp[0] 
            y = tmp[1]
            net.train()
            X = X_.to(device)
            y = y.to(device)
            
            time2 = time.time()
            y_binary, symbol_batch = get_batch_fast(y, net.symbol_set, negative = True, p = args.p)
            with torch.no_grad(): symbol_batch.data += torch.empty(symbol_batch.shape).uniform_(-0.1, 0.1).to(device)
            
            time3 = time.time()
            y_hat = net(X, symbol_batch)
            l = loss(y_hat, y_binary)
            if args.use_orthg: l += net.symbol_orthg()
            
            time4 = time.time()
            param_opti.zero_grad()
            symbol_opti.zero_grad()
            l.backward()
            
            time5 = time.time()
            symbol_opti.step()

            time6 = time.time()
            train_l_sum += l.cpu().item()
            idx = (y_hat.argmax(dim = 1) == y_binary.argmax(dim = 1))
            train_acc_sum += idx.sum().cpu().item()
            n += y.shape[0] if args.cntrst is None else y.shape[0] * (args.cntrst + 1)
            batch_count += 1
            
            cpu_gpu_time += time2 - time1
            make_batch_time += time3 - time2
            forward_time += time4 - time3
            backward_time += time5 - time4
            grad_step_time += time6 - time5
            time7 = time.time()
            other_stat_tims += time7 - time6

        if args.print_frequency != 0 and (epoch + 1) % args.print_frequency == 0:
            time8 = time.time()
            test_acc = evaluate_accuracy(test_set, net, batch_size = args.batch_size) if test_set != None else 0
            time9 = time.time()
            eval_time = time9 - time8
            if logger is None:
                print('epoch %d, concept phase, loss %.4f, train acc %.4f, test acc %.4f, time %.2f sec'
                    % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            else:
                logger.info(f'epoch {epoch + 1}, concept phase, loss {(train_l_sum / batch_count):.4f}, train acc {(train_acc_sum / n):.4f}, test acc given symbol {test_acc:.4f}')
                logger.info(f'total time {(time.time() - start):.3f}sec, IO {cpu_gpu_time:.3f}sec, make batch {make_batch_time:.3f}sec, forward {forward_time:.3f}sec, backward {backward_time:.3f}sec, grad step {grad_step_time:.3f}sec, others stat {other_stat_tims:.3f}sec, eval {eval_time:.3f}sec')
        
        if ckpt is not None:
            if (epoch + 1) % 1 == 0: torch.save(net.state_dict(), ckpt)

def train_concept_first(
    net, 
    loss, 
    train_set, 
    test_set, 
    param_opti, 
    symbol_opti, 
    device, 
    args, 
    logger = None, 
    ckpt = None,
):
    train_iter = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, num_workers = 8, shuffle = True)
    net.to(device)
    batch_ckpt_count = 0

    for epoch in range(args.num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        cpu_gpu_time = 0.0
        make_batch_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        grad_step_time = 0.0
        other_stat_tims = 0.0
        for tmp in tqdm(train_iter):
            time1 = time.time()
            X_ = tmp[0] 
            y = tmp[1]
            net.train()
            X = X_.to(device)
            y = y.to(device)
            
            time2 = time.time()
            y_binary, symbol_batch = get_batch_fast(y, net.symbol_set, negative = True, p = args.p)
            with torch.no_grad(): symbol_batch.data += torch.empty(symbol_batch.shape).uniform_(-0.1, 0.1).to(device)
            
            time3 = time.time()
            y_hat = net(X, symbol_batch)
            l = loss(y_hat, y_binary)
            if args.use_orthg: l += net.symbol_orthg()
            
            time4 = time.time()
            param_opti.zero_grad()
            symbol_opti.zero_grad()
            l.backward()
            
            time5 = time.time()
            symbol_opti.step()

            time6 = time.time()
            train_l_sum += l.cpu().item()
            idx = (y_hat.argmax(dim = 1) == y_binary.argmax(dim = 1))
            train_acc_sum += idx.sum().cpu().item()
            n += y.shape[0] if args.cntrst is None else y.shape[0] * (args.cntrst + 1)
            batch_count += 1
            
            cpu_gpu_time += time2 - time1
            make_batch_time += time3 - time2
            forward_time += time4 - time3
            backward_time += time5 - time4
            grad_step_time += time6 - time5
            time7 = time.time()
            other_stat_tims += time7 - time6

        if args.print_frequency != 0 and (epoch + 1) % args.print_frequency == 0:
            time8 = time.time()
            test_acc = evaluate_accuracy(test_set, net, batch_size = args.batch_size) if test_set != None else 0
            time9 = time.time()
            eval_time = time9 - time8
            if logger is None:
                print('epoch %d, concept phase, loss %.4f, train acc %.4f, test acc %.4f, time %.2f sec'
                    % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            else:
                logger.info(f'epoch {epoch + 1}, concept phase, loss {(train_l_sum / batch_count):.4f}, train acc {(train_acc_sum / n):.4f}, test acc given symbol {test_acc:.4f}')
                logger.info(f'total time {(time.time() - start):.3f}sec, IO {cpu_gpu_time:.3f}sec, make batch {make_batch_time:.3f}sec, forward {forward_time:.3f}sec, backward {backward_time:.3f}sec, grad step {grad_step_time:.3f}sec, others stat {other_stat_tims:.3f}sec, eval {eval_time:.3f}sec')
                
        if not args.random_ts_cdp:
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            cpu_gpu_time = 0.0
            make_batch_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            grad_step_time = 0.0
            other_stat_tims = 0.0
            for tmp in tqdm(train_iter):
                time1 = time.time()
                X_ = tmp[0] 
                y = tmp[1]
                net.train()
                X = X_.to(device)
                y = y.to(device)
                
                time2 = time.time()
                y_binary, symbol_batch = get_batch_fast(y, net.symbol_set, negative = True, p = args.p)
                
                time3 = time.time()
                y_hat = net(X, symbol_batch)
                l = loss(y_hat, y_binary)
                if args.use_orthg: l += net.symbol_orthg()
                
                time4 = time.time()
                param_opti.zero_grad()
                symbol_opti.zero_grad()
                l.backward()
                
                time5 = time.time()
                param_opti.step()

                time6 = time.time()
                train_l_sum += l.cpu().item()
                idx = (y_hat.argmax(dim = 1) == y_binary.argmax(dim = 1))
                train_acc_sum += idx.sum().cpu().item()
                n += y.shape[0] if args.cntrst is None else y.shape[0] * (args.cntrst + 1)
                batch_count += 1
                
                cpu_gpu_time += time2 - time1
                make_batch_time += time3 - time2
                forward_time += time4 - time3
                backward_time += time5 - time4
                grad_step_time += time6 - time5
                time7 = time.time()
                other_stat_tims += time7 - time6

                # torch.save(net.state_dict(), f"./param/sea-net_imagenet1k_ne5_trail5_ckpt{batch_ckpt_count}.pt")
                # batch_ckpt_count += 1

            if args.print_frequency != 0 and (epoch + 1) % args.print_frequency == 0:
                time8 = time.time()
                test_acc = evaluate_accuracy(test_set, net, batch_size = args.batch_size) if test_set != None else 0
                time9 = time.time()
                eval_time = time9 - time8
                if logger is None:
                    print('epoch %d, network training phase, loss %.4f, train acc %.4f, test acc %.4f, time %.2f sec'
                        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
                else:
                    logger.info(f'epoch {epoch + 1}, network phase, loss {(train_l_sum / batch_count):.4f}, train acc {(train_acc_sum / n):.4f}, test acc given symbol {test_acc:.4f}')
                    logger.info(f'total time {(time.time() - start):.3f}sec, IO {cpu_gpu_time:.3f}sec, make batch {make_batch_time:.3f}sec, forward {forward_time:.3f}sec, backward {backward_time:.3f}sec, grad step {grad_step_time:.3f}sec, others stat {other_stat_tims:.3f}sec, eval {eval_time:.3f}sec')
        
        if ckpt is not None:
            if (epoch + 1) % 1 == 0: torch.save(net.state_dict(), ckpt)

def eval_dataload_time(net, loss, train_set, test_set, param_opti, symbol_opti, device, args, logger = None, ckpt = None):
    train_iter = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, num_workers = 4, shuffle = True)
    net.to(device)

    cpu_gpu_time = 0.0
    make_batch_time = 0.0
    for tmp in tqdm(train_iter):
        
        time1 = time.time()
        X = tmp[0].to(device)
        y = tmp[1].to(device)
        
        time2 = time.time()
        
        y_binary, symbol_batch = get_batch_fast(y, net.symbol_set, negative = True, p = args.p)

        time3 = time.time()

        cpu_gpu_time += time2 - time1

        make_batch_time += time3 - time2

    logger.info(f'Transfer tensor from CPU to GPU: {cpu_gpu_time}sec, make batch: {make_batch_time}sec')