"""
Visualization script for attention analysis results.

This script reads JSON results from attn_dist.py and creates visualizations
showing AE and HTA values across layers.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def load_results(json_path: str) -> dict:
    """
    Load attention analysis results from JSON file.
    
    Supports both old format (single prompt) and new format (multiple prompts).
    
    Args:
        json_path: Path to JSON file containing analysis results
    
    Returns:
        Dictionary containing the loaded data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if it's the new format (multiple prompts)
    if 'prompts' in data:
        return data
    # Old format (single prompt) - convert to new format for compatibility
    elif 'prompt' in data and 'layers' in data:
        return {
            'model': data.get('model', 'unknown'),
            'prompts': [{
                'prompt': data['prompt'],
                'layers': data['layers']
            }]
        }
    else:
        raise ValueError("Invalid JSON format: expected 'prompts' or 'prompt'/'layers' keys")


def plot_attention_metrics(data: dict, output_path: Optional[str] = None, show_plot: bool = True):
    """
    Plot AE and HTA values across layers.
    
    Args:
        data: Dictionary containing analysis results
        output_path: Path to save the plot (if None, only display)
        show_plot: Whether to display the plot
    """
    layers = sorted([int(k) for k in data['layers'].keys()])
    ae_values = [data['layers'][str(l)]['ae'] for l in layers]
    hta_values = [data['layers'][str(l)]['hta'] for l in layers]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot AE
    ax1.plot(layers, ae_values, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Attention Entropy (AE)', fontsize=12)
    ax1.set_title('Attention Entropy by Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers[::max(1, len(layers)//10)])  # Show every nth layer label
    
    # Plot HTA
    ax2.plot(layers, hta_values, marker='o', linewidth=2, markersize=6, color='#A23B72')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('HTA Ratio', fontsize=12)
    ax2.set_title('Harmful Token Attention Ratio by Layer', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers[::max(1, len(layers)//10)])
    
    # Add statistics text
    stats_text = f"Prompt: {data.get('prompt', 'N/A')[:50]}...\n"
    stats_text += f"Model: {data.get('model', 'N/A')}\n"
    stats_text += f"Avg AE: {np.mean(ae_values):.4f}\n"
    stats_text += f"Avg HTA: {np.mean(hta_values):.4f}"
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_combined_metrics(data: dict, output_path: Optional[str] = None, show_plot: bool = True):
    """
    Plot AE and HTA on the same plot with dual y-axes.
    
    Args:
        data: Dictionary containing analysis results
        output_path: Path to save the plot (if None, only display)
        show_plot: Whether to display the plot
    """
    layers = sorted([int(k) for k in data['layers'].keys()])
    ae_values = [data['layers'][str(l)]['ae'] for l in layers]
    hta_values = [data['layers'][str(l)]['hta'] for l in layers]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot AE on left y-axis
    color1 = '#2E86AB'
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Attention Entropy (AE)', color=color1, fontsize=12)
    line1 = ax1.plot(layers, ae_values, marker='o', linewidth=2, markersize=6, 
                     color=color1, label='AE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot HTA on right y-axis
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('HTA Ratio', color=color2, fontsize=12)
    line2 = ax2.plot(layers, hta_values, marker='s', linewidth=2, markersize=6, 
                      color=color2, label='HTA')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title
    plt.title('Attention Metrics by Layer', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    lines = line1 + line2
    labels = [str(l.get_label()) for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_correlation(data: dict, output_path: Optional[str] = None, show_plot: bool = True):
    """
    Plot correlation between AE and HTA across layers.
    
    Args:
        data: Dictionary containing analysis results
        output_path: Path to save the plot (if None, only display)
        show_plot: Whether to display the plot
    """
    layers = sorted([int(k) for k in data['layers'].keys()])
    ae_values = [data['layers'][str(l)]['ae'] for l in layers]
    hta_values = [data['layers'][str(l)]['hta'] for l in layers]
    
    # Calculate correlation
    correlation = np.corrcoef(ae_values, hta_values)[0, 1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(ae_values, hta_values, c=layers, cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Add layer labels
    for i, layer in enumerate(layers):
        ax.annotate(f'L{layer}', (ae_values[i], hta_values[i]), 
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Attention Entropy (AE)', fontsize=12)
    ax.set_ylabel('HTA Ratio', fontsize=12)
    ax.set_title(f'AE vs HTA Correlation (r={correlation:.3f})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Layer', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_safe_filename(prompt: str, max_length: int = 50) -> str:
    """Create a safe filename from prompt text."""
    safe_name = "".join(c for c in prompt[:max_length] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    return safe_name


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention analysis results from attn_dist.py"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to JSON file containing analysis results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory to save plots (default: outputs/graphs). A subdirectory with JSON filename will be created."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["separate", "combined", "correlation", "all"],
        default="separate",
        help="Visualization mode: separate plots, combined plot, correlation, or all"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots (default: only save to files)"
    )
    
    args = parser.parse_args()
    
    # Load results
    if not Path(args.input_json).exists():
        print(f"Error: File {args.input_json} not found.")
        return
    
    data = load_results(args.input_json)
    
    # Determine output directory structure
    json_path = Path(args.input_json)
    json_stem = json_path.stem  # filename without extension
    
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path("outputs/graphs")
    
    # Create subdirectory with JSON filename
    output_dir = base_output_dir / json_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get prompts list
    prompts_data = data.get('prompts', [])
    if not prompts_data:
        print("Error: No prompts found in JSON file.")
        return
    
    print(f"Found {len(prompts_data)} prompt(s) in the results file.")
    print(f"Generating visualizations in: {output_dir}\n")
    
    # Generate visualizations for each prompt
    show_plot = args.show
    
    for idx, prompt_data in enumerate(prompts_data, 1):
        prompt = prompt_data['prompt']
        prompt_results = {
            'prompt': prompt,
            'model': data.get('model', 'unknown'),
            'layers': prompt_data['layers']
        }
        
        safe_name = create_safe_filename(prompt)
        print(f"Processing prompt {idx}/{len(prompts_data)}: {prompt[:50]}...")
        
        # Generate visualizations based on mode
        if args.mode == "separate" or args.mode == "all":
            output_path = output_dir / f"{idx:03d}_{safe_name}_separate.png"
            plot_attention_metrics(prompt_results, output_path=str(output_path), show_plot=show_plot)
            print(f"  ✓ Saved: {output_path.name}")
        
        if args.mode == "combined" or args.mode == "all":
            output_path = output_dir / f"{idx:03d}_{safe_name}_combined.png"
            plot_combined_metrics(prompt_results, output_path=str(output_path), show_plot=show_plot)
            print(f"  ✓ Saved: {output_path.name}")
        
        if args.mode == "correlation" or args.mode == "all":
            output_path = output_dir / f"{idx:03d}_{safe_name}_correlation.png"
            plot_correlation(prompt_results, output_path=str(output_path), show_plot=show_plot)
            print(f"  ✓ Saved: {output_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Completed! All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
