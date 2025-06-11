import torch
import torch.nn as nn
import copy


def generate_train_id(num_class, test_ids):
    train_idxs = list(range(num_class))
    for test_id in test_ids:
        train_idxs.remove(test_id)
    return train_idxs

def generate_negative_samp_id(id_list, test_ids):
    neg_samp_id = copy.deepcopy(id_list)
    for test_id in test_ids:
        neg_samp_id.remove(test_id)
    return neg_samp_id

def print_args(parser_args, file_info):
    for arg in vars(parser_args):
        print(arg, getattr(parser_args, arg), file=file_info)

def save_model(path_name_whole, path_name_para_only, net):
    torch.save(net, path_name_whole) 
    torch.save(net.state_dict(), path_name_para_only)

def get_location(tot_num, train_idx, test_idx):
    id_location = list(range(tot_num))
    for train_id in train_idx:
        id_location[train_id] = train_idx.index(train_id)
    for s, test_id in enumerate(test_idx):
        id_location[test_id] = tot_num-s-1
    return id_location

def save_results_mix(epoch, start_epoch, loss, acc, file_haddle, file_type):
    if file_type=="plainText":
        if epoch==start_epoch:
            print('Epoch, Loss, Accuracy', file=file_haddle)
        print('%d, %.3f, %.3f' % (epoch, loss, acc), file=file_haddle)
    if file_type=="markDown":
        if epoch==start_epoch:
            print('| Epoch | Loss | Accuracy |', file=file_haddle)
            print('| ---- | ---- | ---- |', file=file_haddle)
        print('| %d | %.3f | %.3f |' % (epoch, loss, acc), file=file_haddle)

def save_results_mix_pn(epoch, start_epoch, loss_pos, acc_pos, loss_neg, acc_neg, file_haddle, file_type):
    if file_type=="plainText":
        if epoch==start_epoch:
            print('Epoch, Loss_pos, Accuracy_pos, Loss_neg, Accuracy_neg', file=file_haddle)
        print('%d, %.3f, %.3f, %.3f, %.3f' % (epoch, loss_pos, acc_pos, loss_neg, acc_neg), file=file_haddle)
    if file_type=="markDown":
        if epoch==start_epoch:
            print('| Epoch | Loss_pos | Accuracy_pos | Loss_neg | Accuracy_neg |', file=file_haddle)
            print('| ---- | ---- | ---- | ---- | ---- |', file=file_haddle)
        print('| %d | %.3f | %.3f | %.3f | %.3f |' % (epoch, loss_pos, acc_pos, loss_neg, acc_neg), file=file_haddle)

def save_results_seperate_pn(epoch, acc_num, tot_num, context_id, sample_ids, file_haddle, file_type, print_head=False):
    if file_type=="plainText":
        if print_head:
            print('Epoch, Context_id, Sample_id, Accuracy', file=file_haddle)
        for loc, sample_id in enumerate(sample_ids):
            print('%d, %d, %d, %.3f' % (epoch, context_id, sample_id, acc_num[loc]/tot_num[loc]), file=file_haddle)
    if file_type=="markDown":
        if print_head:
            print('| Epoch | Context_id| Sample_id | Accuracy |', file=file_haddle)
            print('| ---- | ---- | ---- | ---- |', file=file_haddle)
        for loc, sample_id in enumerate(sample_ids):
            print('| %d | %d | %d | %.3f |' %  (epoch, context_id, sample_id, acc_num[loc]/tot_num[loc]), file=file_haddle)

def label_translator(origianl_labels, labels_order):
    new_labels = torch.empty(origianl_labels.size(), dtype=torch.long)
    for loc, label in enumerate(labels_order):
        new_labels[origianl_labels==label] = loc
    return new_labels