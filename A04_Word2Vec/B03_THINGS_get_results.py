#!/usr/bin/env python3
"""
Training Results Analysis Script

This script analyzes training log files from THINGS experiments and provides statistics
on completed training runs including accuracy metrics.
"""

import os
import re
import glob
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional


def extract_final_accuracy(log_file_path: str) -> Optional[float]:
    """
    Extract the final best accuracy from a training log file.
    
    Args:
        log_file_path: Path to the training log file
        
    Returns:
        Final accuracy as float, or None if not found or incomplete
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check if training is completed
        if not lines or "=== Training Completed ===" not in lines[-1].strip():
            return None
        
        # Look for the final accuracy line (should be second to last line)
        for line in reversed(lines[:-1]):  # Skip the last line (Training Completed)
            if "Final best accuracy for test class" in line:
                # Extract accuracy using regex
                match = re.search(r'Final best accuracy for test class \d+: ([\d.]+)%', line)
                if match:
                    return float(match.group(1))
        
        return None
        
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None


def analyze_experiment_folder(folder_path: str) -> Dict:
    """
    Analyze all training logs in an experiment folder.
    
    Args:
        folder_path: Path to the experiment folder
        
    Returns:
        Dictionary with analysis results
    """
    folder_name = os.path.basename(folder_path)
    log_files = glob.glob(os.path.join(folder_path, "training_log_class_*.log"))
    
    completed_logs = []
    incomplete_logs = []
    accuracies = []
    
    for log_file in log_files:
        accuracy = extract_final_accuracy(log_file)
        if accuracy is not None:
            completed_logs.append(os.path.basename(log_file))
            accuracies.append(accuracy)
        else:
            incomplete_logs.append(os.path.basename(log_file))
    
    # Calculate statistics
    stats = {}
    if accuracies:
        stats['mean'] = np.mean(accuracies)
        stats['std'] = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
        stats['variance'] = np.var(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
        stats['min'] = np.min(accuracies)
        stats['max'] = np.max(accuracies)
        stats['median'] = np.median(accuracies)
    
    return {
        'folder_name': folder_name,
        'total_logs': len(log_files),
        'completed_count': len(completed_logs),
        'incomplete_count': len(incomplete_logs),
        'completed_logs': completed_logs,
        'incomplete_logs': incomplete_logs,
        'accuracies': accuracies,
        'statistics': stats
    }


def main():
    """Main function to analyze all experiment folders."""
    
    results_dir = "/workspace/A04_Word2Vec/THINGS_results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all experiment folders
    experiment_folders = [
        os.path.join(results_dir, d) 
        for d in os.listdir(results_dir) 
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    
    experiment_folders.sort()
    
    print("=" * 80)
    print("THINGS Training Results Analysis")
    print("=" * 80)
    print(f"Found {len(experiment_folders)} experiment folders")
    print()
    
    all_accuracies = []
    total_completed = 0
    total_logs = 0
    
    # Analyze each experiment folder
    for folder_path in experiment_folders:
        result = analyze_experiment_folder(folder_path)
        
        print(f"📁 Experiment: {result['folder_name']}")
        print(f"   Total logs: {result['total_logs']}")
        print(f"   ✅ Completed: {result['completed_count']}")
        print(f"   ⏳ Incomplete: {result['incomplete_count']}")
        
        if result['accuracies']:
            stats = result['statistics']
            print(f"   📊 Accuracy Statistics:")
            print(f"      Mean: {stats['mean']:.2f}%")
            print(f"      Std:  {stats['std']:.2f}%")
            print(f"      Min:  {stats['min']:.2f}%")
            print(f"      Max:  {stats['max']:.2f}%")
            print(f"      Median: {stats['median']:.2f}%")
            
            all_accuracies.extend(result['accuracies'])
        
        if result['incomplete_logs']:
            print(f"   ❌ Incomplete logs: {', '.join(result['incomplete_logs'][:3])}")
            if len(result['incomplete_logs']) > 3:
                print(f"      ... and {len(result['incomplete_logs']) - 3} more")
        
        total_completed += result['completed_count']
        total_logs += result['total_logs']
        print()
    
    # Overall statistics
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total experiment folders: {len(experiment_folders)}")
    print(f"Total log files: {total_logs}")
    print(f"Total completed training runs: {total_completed}")
    print(f"Total incomplete training runs: {total_logs - total_completed}")
    print(f"Completion rate: {(total_completed / total_logs * 100):.1f}%")
    
    if all_accuracies:
        print()
        print("📊 COMBINED ACCURACY STATISTICS")
        print(f"   Sample size: {len(all_accuracies)} completed runs")
        print(f"   Mean accuracy: {np.mean(all_accuracies):.2f}%")
        print(f"   Standard deviation: {np.std(all_accuracies, ddof=1):.2f}%")
        print(f"   Variance: {np.var(all_accuracies, ddof=1):.2f}")
        print(f"   Min accuracy: {np.min(all_accuracies):.2f}%")
        print(f"   Max accuracy: {np.max(all_accuracies):.2f}%")
        print(f"   Median accuracy: {np.median(all_accuracies):.2f}%")
        
        # Percentiles
        print(f"   25th percentile: {np.percentile(all_accuracies, 25):.2f}%")
        print(f"   75th percentile: {np.percentile(all_accuracies, 75):.2f}%")
    
    print("=" * 80)


if __name__ == "__main__":
    main()