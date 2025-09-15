import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import torchvision.models as models
import scipy.io as io
import os
from pathlib import Path
import argparse
import scipy.stats as stats
import random
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parent.parent
sys.path.append(os.path.join(project_root, "Deps", "CustomFuctions"))

import MixDataLoader, SeparatedDataLoader
import SEAnet

# parameters
parser = argparse.ArgumentParser(description='set parameters')
parser.add_argument('--dataset_path', type=str, default='/data/THINGS/object_images', help='path to THINGS dataset')
parser.add_argument('--embedding_path', type=str, default='/data/THINGS/spose_embedding_49d_sorted.txt', help='path to spose embedding file')
parser.add_argument('--device', type=str, default='cuda', help='cuda if torch.cuda.is_available() else cpu')
parser.add_argument('--worker', type=int, default=4, help='dataloader num_workers')
parser.add_argument('--batch_size_train', type=int, default=1024, help='batch size for training dataset')
parser.add_argument('--batch_size_test', type=int, default=128, help='batch size for testing dataset')
parser.add_argument('--num_classes_total', type=int, default=1854, help='total number of classes in THINGS dataset')
parser.add_argument('--num_classes_experiment', type=int, default=100, help='number of classes to use in experiment')
parser.add_argument('--test_id', type=int, nargs='+', default=None, help='the ids of the classes used to test')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number to end with')
parser.add_argument('--context_dim', type=int, default=49, help='context dimension (embedding dimension)')
parser.add_argument('--noise_intensity', type=float, nargs='+', default=[0.1], help='intensity of noises added to the contexts')
parser.add_argument('--train_mode', type=bool, default=False, help='set TRUE to train the network, FALSE to test the network')
parser.add_argument('--drop_probility', type=float, default=0.2, help='drop probility for CDP network 3 to avoid over-fitting')
parser.add_argument('--num_process', type=int, default=10, help='how many processes to run in parallel')
parser.add_argument('--random_seed', type=int, default=None, help='random seed for reproducibility')
args = parser.parse_args()

# Set random seeds for reproducibility
if args.random_seed is not None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

# Load embedding vectors and class names
print('==> Loading embedding vectors and class names..')
embedding_vectors = MixDataLoader.load_spose_embedding(args.embedding_path)
class_names = MixDataLoader.get_things_class_names(args.dataset_path)
print(f'Total classes: {len(class_names)}, Embedding dimension: {embedding_vectors.shape[1]}')

# Randomly select classes for testing if not specified
if args.test_id is None:
    selected_test_class_indices = random.sample(range(len(class_names)), args.num_classes_experiment)
    args.test_id = selected_test_class_indices
else:
    selected_test_class_indices = args.test_id

print(f'Selected {len(selected_test_class_indices)} classes for leave-one-out testing')
print(f'Training will use all other {len(class_names) - 1} classes for each test class')

def get_batch_fast(y, symbol_set, negative = False, p = 0.5):
    device = symbol_set.device
    batch_size = len(y)
    num_symbols = symbol_set.shape[0] # 1854

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

def train(epoch, ni, trainloader_things, my_extended_model, optimizer_net, contexts, criterion):
    """
    Training function for one epoch with detailed timing statistics
    """
    import time
    print('\nEpoch: %d' % epoch)
    my_extended_model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    start = time.time()
    
    # Initialize timing variables
    cpu_gpu_time = 0.0
    context_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    grad_step_time = 0.0
    other_stat_time = 0.0
    
    for (inputs, targets) in tqdm(trainloader_things):
        time1 = time.time()
        
        # Move data to device
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        time2 = time.time()
        
        # Map targets to training indices and get contexts
        y_binary, symbol_batch = get_batch_fast(targets, contexts[0], negative = True, p = 0.5)
        with torch.no_grad(): symbol_batch.data += torch.empty(symbol_batch.shape).uniform_(-ni, ni).to(args.device)

        time3 = time.time()
        
        # Forward pass
        y_hat = my_extended_model(inputs, symbol_batch)
        
        # Create target tensor with correct batch size
        loss = criterion(y_hat, y_binary)
        
        time4 = time.time()
        
        # Backward pass
        optimizer_net.zero_grad()
        loss.backward()
        
        time5 = time.time()
        
        # Optimizer step
        optimizer_net.step()
        
        time6 = time.time()
        
        # Statistics calculation
        train_loss += loss.cpu().item()
        idx = (y_hat.argmax(dim = 1) == y_binary.argmax(dim = 1))
        correct += idx.sum().cpu().item()
        total += targets.shape[0]
        batch_count += 1
        
        # Update timing statistics
        cpu_gpu_time += time2 - time1
        context_time += time3 - time2
        forward_time += time4 - time3
        backward_time += time5 - time4
        grad_step_time += time6 - time5
        time7 = time.time()
        other_stat_time += time7 - time6
        
        # Print detailed timing for each batch
        batch_time = time7 - time1
        data_time = time2 - time1
        batch_forward_time = time4 - time3
        batch_backward_time = time5 - time4
        optimizer_time = time6 - time5
        
        print(f'Batch {batch_count}/{len(trainloader_things)} - '
              f'Total: {batch_time:.4f}s, '
              f'Data: {data_time:.4f}s ({data_time/batch_time*100:.1f}%), '
              f'Forward: {batch_forward_time:.4f}s ({batch_forward_time/batch_time*100:.1f}%), '
              f'Backward: {batch_backward_time:.4f}s ({batch_backward_time/batch_time*100:.1f}%), '
              f'Optim: {optimizer_time:.4f}s ({optimizer_time/batch_time*100:.1f}%), '
              f'Train Loss: {loss.item():.6f}, '
              f'Train Acc: {100. * correct / total:.2f}%')
    
    total_time = time.time() - start
    avg_loss = train_loss / batch_count
    avg_acc = 100. * correct / total
    
    print(f'Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%')
    print(f'Timing Statistics:')
    print(f'  Total time: {total_time:.3f}sec')
    print(f'  CPU->GPU transfer: {cpu_gpu_time:.3f}sec ({100*cpu_gpu_time/total_time:.1f}%)')
    print(f'  Context preparation: {context_time:.3f}sec ({100*context_time/total_time:.1f}%)')
    print(f'  Forward pass: {forward_time:.3f}sec ({100*forward_time/total_time:.1f}%)')
    print(f'  Backward pass: {backward_time:.3f}sec ({100*backward_time/total_time:.1f}%)')
    print(f'  Gradient step: {grad_step_time:.3f}sec ({100*grad_step_time/total_time:.1f}%)')
    print(f'  Other statistics: {other_stat_time:.3f}sec ({100*other_stat_time/total_time:.1f}%)')
    print(f'  Average time per batch: {total_time/batch_count:.4f}sec')
    
    return avg_loss, avg_acc, total_time, {
        'cpu_gpu_time': cpu_gpu_time,
        'context_time': context_time,
        'forward_time': forward_time,
        'backward_time': backward_time,
        'grad_step_time': grad_step_time,
        'other_stat_time': other_stat_time,
        'total_time': total_time,
        'batch_count': batch_count
    }

def test(testloader_things, my_extended_model, contexts, criterion):
    """
    Testing function
    """
    my_extended_model.eval()
    test_loss = 0
    total = 0
    correct = 0
    batch_count = 0
    
    with torch.no_grad():
        for (inputs, targets) in tqdm(testloader_things):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            y_binary, symbol_batch = get_batch_fast(targets, contexts[0], negative = True, p = 0.5)
            
            y_hat = my_extended_model(inputs, symbol_batch)
            idx = (y_hat.argmax(dim = 1) == y_binary.argmax(dim = 1))
            
            correct += idx.sum().cpu().item()
            total += targets.shape[0]
            loss = criterion(y_hat, y_binary)
            test_loss += loss.cpu().item()
            batch_count += 1
    
    acc = 100. * (correct) / (total)
    avg_test_loss = test_loss / (batch_count)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {acc:.2f}%')
    return avg_test_loss, acc

run_time = datetime.now().date()

def model_training(test_id, ni, timestamp):
    """
    Main training function for leave-one-out experiment
    """
    # Use all classes except the test class for training
    train_id = [x for x in range(len(class_names)) if x not in [test_id]]
    all_id = train_id + [test_id]
    best_acc = 0
    
    print(f'\n==> Training for test class {test_id} (class name: {class_names[test_id]})')
    print(f'Training classes: {len(train_id)} (all except test class), Test class: 1')
    
    # Import data using optimized loaders
    print('==> Preparing data with optimized loaders..')
    trainloader_things = MixDataLoader.DLTrain_things_part_optimized(args.dataset_path, train_id, args.batch_size_train, args.worker)
    testloader_things = MixDataLoader.DLTest_things_part_optimized(args.dataset_path, [test_id], args.batch_size_test, args.worker)
    print(f'Training dataset size: {len(trainloader_things.dataset)} samples')
    print(f'Test dataset size: {len(testloader_things.dataset)} samples')
    
    # Context initialization - use all 1854 classes
    contexts = [torch.zeros(len(class_names), args.context_dim).to(args.device)]
    for class_idx in range(len(class_names)):
        contexts[0][class_idx, :] = torch.tensor(embedding_vectors[class_idx], dtype=torch.float).to(args.device)
    
    # Save initial context
    context_path = script_dir / 'THINGS_results' / f'{timestamp}_ni={ni:.2e}' / 'contexts' / f'context_initial_id_{test_id}.mat'
    io.savemat(str(context_path),
               {'contexts': np.array(contexts[0].cpu().detach()),
                'all_idx': np.array(all_id),
                'trained_idx': np.array(train_id),
                'tested_idx': np.array(test_id)})
    
    # Load pretrained model
    print('==> Loading pretrained model..')
    pretrained_classifier_cnn = models.resnet18()
    pretrained_classifier_cnn.load_state_dict(
        torch.load(
            os.path.join(project_root, "Deps", "pretrained_fe", "resnet18-f37072fd.pth")
        )
    )
    pretrained_classifier_cnn.fc = nn.Identity()
    
    # Structure and parameter setting
    my_extended_model = SEAnet.Net2(
        my_pretrained_classifier=pretrained_classifier_cnn,
        context_dim=args.context_dim
    ).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer_net = optim.Adam(my_extended_model.parameters(), lr=0.0001)
    
    for epoch in range(args.start_epoch, args.end_epoch):
        avg_loss, avg_acc, total_time, timing_stats = train(epoch, ni, trainloader_things, my_extended_model, optimizer_net, contexts, criterion)
        
        # Test after each epoch
        test_loss, test_acc = test(testloader_things, my_extended_model, contexts, criterion)
        
        # Save checkpoint if this is the best accuracy so far
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_path = script_dir / 'THINGS_results' / f'{timestamp}_ni={ni:.2e}' / 'checkpoint' / f'best_model_id_{test_id}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': my_extended_model.state_dict(),
                'optimizer_state_dict': optimizer_net.state_dict(),
                'best_acc': best_acc,
                'test_loss': test_loss,
                'contexts': contexts[0]
            }, str(checkpoint_path))
            print(f'New best accuracy: {best_acc:.4f}%, model saved to {checkpoint_path}')
        
        print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%, Best Acc: {best_acc:.4f}%')
        
        # Print detailed timing statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n=== Epoch {epoch} Training Time Statistics ===")
            print(f"Total epoch time: {timing_stats['total_time']:.4f}s")
            print(f"CPU->GPU transfer time: {timing_stats['cpu_gpu_time']:.4f}s ({timing_stats['cpu_gpu_time']/timing_stats['total_time']*100:.1f}%)")
            print(f"Context preparation time: {timing_stats['context_time']:.4f}s ({timing_stats['context_time']/timing_stats['total_time']*100:.1f}%)")
            print(f"Forward pass time: {timing_stats['forward_time']:.4f}s ({timing_stats['forward_time']/timing_stats['total_time']*100:.1f}%)")
            print(f"Backward pass time: {timing_stats['backward_time']:.4f}s ({timing_stats['backward_time']/timing_stats['total_time']*100:.1f}%)")
            print(f"Gradient step time: {timing_stats['grad_step_time']:.4f}s ({timing_stats['grad_step_time']/timing_stats['total_time']*100:.1f}%)")
            print(f"Other statistics time: {timing_stats['other_stat_time']:.4f}s ({timing_stats['other_stat_time']/timing_stats['total_time']*100:.1f}%)")
            print(f"Average batch time: {timing_stats['total_time']/timing_stats['batch_count']:.4f}s")
            print(f"Batches per second: {timing_stats['batch_count']/timing_stats['total_time']:.2f}")
            print("=" * 50)
        
        print(f"Epoch {epoch} completed - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%, Best Test Acc: {best_acc:.2f}%")
    
    print(f'Final best accuracy for test class {test_id}: {best_acc:.2f}%')
    return best_acc

def things49dim_test(timestamp):
    """
    Load trained models and perform testing for each test_id
    """
    print('==> Starting THINGS 49 dim testing..')
    
    for noise_intensity in args.noise_intensity:
        path = script_dir / 'THINGS_results' / f'{timestamp}_ni={noise_intensity:.2e}'
        print(f'Testing with noise intensity: {noise_intensity}')
        final_acc_list = []
        
        for test_id in selected_test_class_indices:
            print(f'Testing class ID: {test_id} (class name: {class_names[test_id]})')
            
            # Prepare checkpoint path
            checkpoint_path = path / 'checkpoint' / f'ckpt_dim_{args.context_dim}_id_{test_id}.pth'
            
            if os.path.exists(str(checkpoint_path)):
                # Load checkpoint
                checkpoint = torch.load(str(checkpoint_path))
                
                # Recreate model
                my_extended_model = SEAnet.Net2(
                    my_pretrained_classifier=pretrained_classifier_cnn,
                    context_dim=args.context_dim
                ).to(args.device)
                my_extended_model.load_state_dict(checkpoint['net'])
                
                # Test
                testloader_things = MixDataLoader.DLTest_things_part_optimized(args.dataset_path, [test_id], args.batch_size_test, args.worker)
                contexts = checkpoint['context']
                id_location = checkpoint['class_index']
                
                criterion = nn.CrossEntropyLoss()
                acc = test(0, testloader_things, my_extended_model, contexts, criterion, None, test_id, id_location)
                final_acc_list.append(acc)
                print(f'Class {test_id} accuracy: {acc:.2f}%')
            else:
                print(f'Checkpoint not found for class {test_id}')
                final_acc_list.append(0.0)
        
        # Calculate statistics
        mean_acc = np.mean(final_acc_list)
        std_acc = np.std(final_acc_list)
        print(f'\nFinal Results for noise intensity {noise_intensity}:')
        print(f'Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%')
        print(f'Individual accuracies: {final_acc_list}')
        
        # Save results
        results = {
            'final_acc_list': final_acc_list,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'selected_test_classes': selected_test_class_indices,
            'test_class_names': [class_names[i] for i in selected_test_class_indices]
        }
        results_path = path / 'final_results.mat'
        io.savemat(str(results_path), results)

# Main execution
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for noise_intensity in args.noise_intensity:
    path = script_dir / 'THINGS_results' / f'{run_timestamp}_ni={noise_intensity:.2e}'
    if not os.path.isdir(path / 'contexts'):
        os.makedirs(path / 'contexts')
    if not os.path.isdir(path / 'checkpoint'):
        os.makedirs(path / 'checkpoint')
    
    # Save experiment metadata
    metadata = {
        'timestamp': run_timestamp,
        'experiment_config': {
            'dataset_path': args.dataset_path,
            'embedding_path': args.embedding_path,
            'num_classes_total': args.num_classes_total,
            'num_classes_experiment': args.num_classes_experiment,
            'selected_test_class_indices': selected_test_class_indices,
            'selected_test_class_names': [class_names[i] for i in selected_test_class_indices],
            'context_dim': args.context_dim,
            'noise_intensity': noise_intensity,
            'batch_size_train': args.batch_size_train,
            'batch_size_test': args.batch_size_test,
            'start_epoch': args.start_epoch,
            'end_epoch': args.end_epoch,
            'num_process': args.num_process,
            'random_seed': args.random_seed,
            'train_mode': args.train_mode,
            'drop_probility': args.drop_probility
        }
    }
    
    with open(str(path / 'experiment_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f'\n==> Starting experiment with noise intensity: {noise_intensity}')
    print(f'Selected test classes: {selected_test_class_indices}')
    print(f'Test class names: {[class_names[i] for i in selected_test_class_indices]}')
    print(f'Each test class will be trained against all other {len(class_names)-1} classes')
    
    if __name__ == '__main__':
        if args.train_mode:
            for test_class_id in selected_test_class_indices:
                # Create log file for this test class
                log_filename = path / f'training_log_class_{test_class_id}_{class_names[test_class_id]}.log'
                
                # Redirect stdout to log file
                original_stdout = sys.stdout
                with open(log_filename, 'w', encoding='utf-8') as log_file:
                    sys.stdout = log_file
                    
                    print(f"=== Training Log for Test Class {test_class_id}: {class_names[test_class_id]} ===")
                    print(f"Timestamp: {run_timestamp}")
                    print(f"Noise Intensity: {noise_intensity}")
                    print(f"Training against {len(class_names)-1} other classes")
                    print("=" * 80)
                    
                    try:
                        model_training(test_class_id, noise_intensity, run_timestamp)
                    except Exception as e:
                        print(f"Error during training: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        print("\n=== Training Completed ===")
                        sys.stdout = original_stdout
                
                # Print to console that log was created
                print(f"Training log for class {test_class_id} ({class_names[test_class_id]}) saved to: {log_filename}")
        else:
            things49dim_test(run_timestamp)
