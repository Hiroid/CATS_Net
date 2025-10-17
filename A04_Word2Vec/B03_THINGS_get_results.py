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
import pandas as pd
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


def create_detailed_results_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """
    Create a detailed DataFrame from all experiment results.
    
    Args:
        all_results: List of experiment analysis results
        
    Returns:
        DataFrame with detailed results for each experiment and training run
    """
    detailed_rows = []
    
    for result in all_results:
        experiment_name = result['folder_name']
        
        # Add rows for each completed training run
        for i, (log_file, accuracy) in enumerate(zip(result['completed_logs'], result['accuracies'])):
            # Extract class ID from log filename if possible
            class_match = re.search(r'class_(\d+)', log_file)
            class_id = int(class_match.group(1)) if class_match else i + 1
            
            detailed_rows.append({
                'experiment_name': experiment_name,
                'log_file': log_file,
                'class_id': class_id,
                'accuracy': accuracy,
                'status': 'completed'
            })
        
        # Add rows for incomplete training runs
        for log_file in result['incomplete_logs']:
            class_match = re.search(r'class_(\d+)', log_file)
            class_id = int(class_match.group(1)) if class_match else None
            
            detailed_rows.append({
                'experiment_name': experiment_name,
                'log_file': log_file,
                'class_id': class_id,
                'accuracy': None,
                'status': 'incomplete'
            })
    
    return pd.DataFrame(detailed_rows)


def create_summary_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """
    Create a summary DataFrame with statistics for each experiment.
    
    Args:
        all_results: List of experiment analysis results
        
    Returns:
        DataFrame with summary statistics for each experiment
    """
    summary_rows = []
    
    for result in all_results:
        stats = result.get('statistics', {})
        
        summary_rows.append({
            'experiment_name': result['folder_name'],
            'total_logs': result['total_logs'],
            'completed_count': result['completed_count'],
            'incomplete_count': result['incomplete_count'],
            'completion_rate': result['completed_count'] / result['total_logs'] * 100 if result['total_logs'] > 0 else 0,
            'mean_accuracy': stats.get('mean', None),
            'std_accuracy': stats.get('std', None),
            'variance_accuracy': stats.get('variance', None),
            'min_accuracy': stats.get('min', None),
            'max_accuracy': stats.get('max', None),
            'median_accuracy': stats.get('median', None)
        })
    
    return pd.DataFrame(summary_rows)


def save_results_to_files(all_results: List[Dict], output_dir: str = "/workspace/Results"):
    """
    Save analysis results to CSV and Excel files.
    
    Args:
        all_results: List of experiment analysis results
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed and summary DataFrames
    detailed_df = create_detailed_results_dataframe(all_results)
    summary_df = create_summary_dataframe(all_results)
    
    # Calculate overall statistics
    all_accuracies = []
    total_completed = 0
    total_logs = 0
    
    for result in all_results:
        all_accuracies.extend(result['accuracies'])
        total_completed += result['completed_count']
        total_logs += result['total_logs']
    
    # Create overall summary
    overall_stats = {}
    if all_accuracies:
        overall_stats = {
            'total_experiments': len(all_results),
            'total_logs': total_logs,
            'total_completed': total_completed,
            'total_incomplete': total_logs - total_completed,
            'overall_completion_rate': total_completed / total_logs * 100 if total_logs > 0 else 0,
            'overall_mean_accuracy': np.mean(all_accuracies),
            'overall_std_accuracy': np.std(all_accuracies, ddof=1) if len(all_accuracies) > 1 else 0.0,
            'overall_variance_accuracy': np.var(all_accuracies, ddof=1) if len(all_accuracies) > 1 else 0.0,
            'overall_min_accuracy': np.min(all_accuracies),
            'overall_max_accuracy': np.max(all_accuracies),
            'overall_median_accuracy': np.median(all_accuracies),
            'overall_25th_percentile': np.percentile(all_accuracies, 25),
            'overall_75th_percentile': np.percentile(all_accuracies, 75),
            'sample_size': len(all_accuracies)
        }
    
    overall_df = pd.DataFrame([overall_stats])
    
    # Save to CSV files
    detailed_csv_path = os.path.join(output_dir, "THINGS_detailed_results.csv")
    summary_csv_path = os.path.join(output_dir, "THINGS_summary_results.csv")
    overall_csv_path = os.path.join(output_dir, "THINGS_overall_stats.csv")
    
    detailed_df.to_csv(detailed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    overall_df.to_csv(overall_csv_path, index=False)
    
    # Save to Excel file with multiple sheets
    excel_path = os.path.join(output_dir, "THINGS_results.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_by_Experiment', index=False)
        overall_df.to_excel(writer, sheet_name='Overall_Statistics', index=False)
    
    print(f"\n📁 Results saved to:")
    print(f"   📄 Detailed results: {detailed_csv_path}")
    print(f"   📄 Summary by experiment: {summary_csv_path}")
    print(f"   📄 Overall statistics: {overall_csv_path}")
    print(f"   📊 Excel file (all sheets): {excel_path}")
    
    return {
        'detailed_csv': detailed_csv_path,
        'summary_csv': summary_csv_path,
        'overall_csv': overall_csv_path,
        'excel_file': excel_path
    }


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
    all_results = []  # Store all results for saving to files
    
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
        all_results.append(result)  # Store result for file export
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
    
    # Save results to files
    if all_results:
        print("\n💾 SAVING RESULTS TO FILES")
        print("=" * 80)
        saved_files = save_results_to_files(all_results)
        print("=" * 80)


if __name__ == "__main__":
    main()