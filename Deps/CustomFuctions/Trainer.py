import torch
import torch.nn as nn
import torch.optim as optim
import Utiliz
def train_model_context(round, epoch, ni, args, trainloader_cifar100, target_extend, optimizer_input, optimizer_net, contexts, my_extended_model, criterion):
    my_extended_model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader_cifar100):
            p = torch.rand(1)
            neg_mask = torch.rand(args.batch_size_train).ge(p[0])
            binary_labels = torch.ones(args.batch_size_train, dtype=torch.long)
            binary_labels[neg_mask] = 0
            rand_num = torch.ceil(torch.rand(neg_mask.sum())*(args.num_class-1)).long()
            context = contexts[0][targets, :]
            context[neg_mask] = contexts[0][target_extend[targets[neg_mask]+rand_num], :]
            context = context + (torch.rand(context.size(), device=args.device)*2.0 - 1.0)*ni
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            context = context.to(args.device)
            binary_labels = binary_labels.to(args.device)

            output = my_extended_model(inputs, context)
            loss = criterion(output, binary_labels)
            optimizer_input.zero_grad()
            optimizer_net.zero_grad()
            loss.backward()
            if epoch%2 == 0:
                optimizer_net.step()
            else:
                optimizer_input.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(binary_labels).sum().item()
    return [train_loss / batch_idx, correct / total]

def train_model_context_part(round, epoch, ni, args, trainloader_cifar100, target_extend, optimizer_input, optimizer_net, contexts, my_extended_model, criterion):
    my_extended_model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets_orig) in enumerate(trainloader_cifar100):
            p = torch.rand(1)
            targets = Utiliz.label_translator(targets_orig, args.train_id)
            neg_mask = torch.rand(targets_orig.size(0)).ge(p[0])
            binary_labels = torch.ones(targets_orig.size(0), dtype=torch.long)
            binary_labels[neg_mask] = 0
            rand_num = torch.ceil(torch.rand(neg_mask.sum())*(args.num_class-1)).long()
            context = contexts[0][targets, :]
            context[neg_mask] = contexts[0][target_extend[targets[neg_mask]+rand_num], :]
            context = context + (torch.rand(context.size(), device=args.device)*2.0 - 1.0)*ni
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            context = context.to(args.device)
            binary_labels = binary_labels.to(args.device)

            output = my_extended_model(inputs, context)
            loss = criterion(output, binary_labels)
            optimizer_input.zero_grad()
            optimizer_net.zero_grad()
            loss.backward()
            if epoch%2 == 0:
                optimizer_net.step()
            else:
                optimizer_input.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(binary_labels).sum().item()
    return [train_loss / batch_idx, correct / total]


