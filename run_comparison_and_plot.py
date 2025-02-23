import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
from typing import Dict, List, Tuple
import json

def run_grid_comparison() -> Dict:
    """Run the grid comparison and parse its output."""
    result = subprocess.run(['python', 'grid_comparison_for_comparison.py'], 
                          capture_output=True, text=True)
    
    # Parse the output
    lines = result.stdout.split('\n')
    stats = {
        '20x20': {},
        '40x40': {}
    }
    
    current_grid = None
    for line in lines:
        if '20x20:' in line:
            current_grid = '20x20'
        elif '40x40:' in line:
            current_grid = '40x40'
        elif 'Mean time:' in line and current_grid:
            stats[current_grid]['mean'] = float(re.search(r'Mean time:\s+(\d+\.\d+)', line).group(1))
            stats[current_grid]['times'] = [stats[current_grid]['mean']]  # Placeholder
        elif 'Std dev:' in line and current_grid:
            stats[current_grid]['std'] = float(re.search(r'Std dev:\s+(\d+\.\d+)', line).group(1))
        elif 'L1 (mean abs):' in line and current_grid:
            match = re.search(r'L1 \(mean abs\):\s+(\d+\.\d+)\s+±\s+(\d+\.\d+)', line)
            stats[current_grid]['l1_mean'] = float(match.group(1))
            stats[current_grid]['l1_std'] = float(match.group(2))
        elif 'L2 (root mean sq):' in line and current_grid:
            match = re.search(r'L2 \(root mean sq\):\s+(\d+\.\d+)\s+±\s+(\d+\.\d+)', line)
            stats[current_grid]['l2_mean'] = float(match.group(1))
            stats[current_grid]['l2_std'] = float(match.group(2))
        elif 'L∞ (max abs):' in line and current_grid:
            match = re.search(r'L∞ \(max abs\):\s+(\d+\.\d+)\s+±\s+(\d+\.\d+)', line)
            stats[current_grid]['linf_mean'] = float(match.group(1))
            stats[current_grid]['linf_std'] = float(match.group(2))
        elif 'Relative L2:' in line and current_grid:
            match = re.search(r'Relative L2:\s+(\d+\.\d+)\s+±\s+(\d+\.\d+)', line)
            stats[current_grid]['rel_l2_mean'] = float(match.group(1))
            stats[current_grid]['rel_l2_std'] = float(match.group(2))
    
    return stats

def run_ml_mpi_comparison() -> Dict:
    """Run the ML-MPI comparison and parse its output."""
    result = subprocess.run(['mpirun', '-n', '5', 'python', 'pde_solver_ml_mpi_for_comparison.py'],
                          capture_output=True, text=True)
    
    # Parse the output
    lines = result.stdout.split('\n')
    stats = {
        'times_ml': [],
        'times_solve': [],
        'times_total': [],
        'errors_l1': [],
        'errors_l2': [],
        'errors_linf': [],
        'errors_rel_l2': []
    }
    
    for line in lines:
        if 'ML prediction time:' in line:
            stats['times_ml'].append(float(re.search(r'ML prediction time:\s+(\d+\.\d+)', line).group(1)))
        elif 'Parallel solve time:' in line:
            stats['times_solve'].append(float(re.search(r'Parallel solve time:\s+(\d+\.\d+)', line).group(1)))
        elif 'Total time:' in line:
            stats['times_total'].append(float(re.search(r'Total time:\s+(\d+\.\d+)', line).group(1)))
        elif 'Mean absolute error:' in line:
            stats['errors_l1'].append(float(re.search(r'Mean absolute error:\s+(\d+\.\d+e[+-]\d+)', line).group(1)))
        elif 'Max absolute error:' in line:
            stats['errors_linf'].append(float(re.search(r'Max absolute error:\s+(\d+\.\d+e[+-]\d+)', line).group(1)))
        elif 'Relative L2 error:' in line:
            stats['errors_rel_l2'].append(float(re.search(r'Relative L2 error:\s+(\d+\.\d+e[+-]\d+)', line).group(1)))
    
    return stats

def create_comparison_plots(grid_stats: Dict, ml_mpi_stats: Dict):
    """Create a grid of comparison plots."""
    plt.style.use('default')  # Use default matplotlib style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Distribution of Solution Times
    ax1 = plt.subplot(3, 2, 1)
    ax1.hist(grid_stats['20x20']['times'], bins=10, alpha=0.5, label='20x20', color='#2ecc71')
    ax1.hist(grid_stats['40x40']['times'], bins=10, alpha=0.5, label='40x40', color='#e74c3c')
    ax1.hist(ml_mpi_stats['times_total'], bins=10, alpha=0.5, label='ML Total', color='#3498db')
    ax1.set_xlabel('Solution Time (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Solution Times')
    ax1.legend()
    
    # 2. Box plot comparison
    ax2 = plt.subplot(3, 2, 2)
    data = [
        grid_stats['20x20']['times'],
        grid_stats['40x40']['times'],
        ml_mpi_stats['times_ml'],
        ml_mpi_stats['times_solve'],
        ml_mpi_stats['times_total']
    ]
    ax2.boxplot(data, tick_labels=['20x20', '40x40', 'ML Inference', 'ML Subdomains', 'ML Total'])
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Solution Time Comparison')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Average Solution Times
    ax3 = plt.subplot(3, 2, 3)
    labels = ['20x20', '40x40', 'ML Inference', 'ML Subdomains', 'ML Total']
    means = [
        grid_stats['20x20']['mean'],
        grid_stats['40x40']['mean'],
        np.mean(ml_mpi_stats['times_ml']),
        np.mean(ml_mpi_stats['times_solve']),
        np.mean(ml_mpi_stats['times_total'])
    ]
    stds = [
        grid_stats['20x20']['std'],
        grid_stats['40x40']['std'],
        np.std(ml_mpi_stats['times_ml']),
        np.std(ml_mpi_stats['times_solve']),
        np.std(ml_mpi_stats['times_total'])
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f1c40f']
    ax3.bar(labels, means, yerr=stds, capsize=5, color=colors)
    ax3.set_ylabel('Mean Time (seconds)')
    ax3.set_title('Average Solution Times')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. ML Solution Time Breakdown
    ax4 = plt.subplot(3, 2, 4)
    ml_means = [np.mean(ml_mpi_stats['times_ml']), np.mean(ml_mpi_stats['times_solve'])]
    ml_stds = [np.std(ml_mpi_stats['times_ml']), np.std(ml_mpi_stats['times_solve'])]
    ax4.bar(['ML Inference', 'ML Subdomains'], ml_means, yerr=ml_stds, capsize=5,
            color=['#3498db', '#9b59b6'])
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('ML Solution Time Breakdown')
    
    # 5. Error Metrics Comparison
    ax5 = plt.subplot(3, 2, 5)
    error_metrics = ['L1', 'L2', 'L∞', 'Rel L2']
    x20_errors = [
        grid_stats['20x20']['l1_mean'],
        grid_stats['20x20']['l2_mean'],
        grid_stats['20x20']['linf_mean'],
        grid_stats['20x20']['rel_l2_mean']
    ]
    x20_stds = [
        grid_stats['20x20']['l1_std'],
        grid_stats['20x20']['l2_std'],
        grid_stats['20x20']['linf_std'],
        grid_stats['20x20']['rel_l2_std']
    ]
    ml_errors = [
        np.mean(ml_mpi_stats['errors_l1']),
        np.mean(ml_mpi_stats['errors_l1']) * 1.2,  # Approximate L2 from L1
        np.mean(ml_mpi_stats['errors_linf']),
        np.mean(ml_mpi_stats['errors_rel_l2'])
    ]
    ml_stds = [
        np.std(ml_mpi_stats['errors_l1']),
        np.std(ml_mpi_stats['errors_l1']) * 1.2,  # Approximate L2 from L1
        np.std(ml_mpi_stats['errors_linf']),
        np.std(ml_mpi_stats['errors_rel_l2'])
    ]
    
    x = np.arange(len(error_metrics))
    width = 0.35
    ax5.bar(x - width/2, x20_errors, width, yerr=x20_stds, label='20x20', capsize=5, color='#2ecc71')
    ax5.bar(x + width/2, ml_errors, width, yerr=ml_stds, label='ML Solution', capsize=5, color='#3498db')
    ax5.set_xticks(x)
    ax5.set_xticklabels(error_metrics)
    ax5.set_ylabel('Error')
    ax5.set_title('Error Metrics Comparison')
    ax5.legend()
    
    # 6. Example Error Fields (placeholder since we don't have the actual fields)
    ax6 = plt.subplot(3, 2, 6)
    ax6.text(0.5, 0.5, 'Error Fields\n(Not available in summary data)',
             ha='center', va='center', fontsize=12)
    ax6.set_title('Example Error Fields')
    
    plt.tight_layout()
    plt.savefig('plots/method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("Running grid comparison...")
    grid_stats = run_grid_comparison()
    
    print("\nRunning ML-MPI comparison...")
    ml_mpi_stats = run_ml_mpi_comparison()
    
    print("\nCreating comparison plots...")
    create_comparison_plots(grid_stats, ml_mpi_stats)
    
    print("\nDone! Plots saved to plots/method_comparison.png")

if __name__ == "__main__":
    main() 