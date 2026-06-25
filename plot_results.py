import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_publication_plots():
    # Set style for academic publication
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    CSV_FILE = "simulation_results.csv"
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: Cannot find '{CSV_FILE}'. Please run simulation.py first.")
        return

    # Load the benchmark results
    df = pd.read_csv(CSV_FILE)
    
    # Split data by controller for easy plotting
    agent_df = df[df['controller'] == 'Agentic_LangGraph'].sort_values('step')
    base_df = df[df['controller'] == 'Baseline_Rule_Based'].sort_values('step')

    # Create a 2x1 grid of highly professional subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # ------------------ Plot 1: Temperature Profile ------------------
    # Baseline
    ax1.plot(base_df['step'], base_df['w_temp'], label='Baseline: Window Zone', 
             color='#ff9999', linestyle='--', marker='o')
    ax1.plot(base_df['step'], base_df['i_temp'], label='Baseline: Interior Zone', 
             color='#9999ff', linestyle='--', marker='^')
    
    # Agentic
    ax1.plot(agent_df['step'], agent_df['w_temp'], label='Agentic: Window Zone', 
             color='#cc0000', linestyle='-', linewidth=2, marker='o')
    ax1.plot(agent_df['step'], agent_df['i_temp'], label='Agentic: Interior Zone', 
             color='#0000cc', linestyle='-', linewidth=2, marker='^')
    
    # Target comfort bands (Visual guidelines)
    ax1.axhline(y=21.8, color='red', linestyle=':', alpha=0.5, label='Window Target (21.8°C)')
    ax1.axhline(y=24.0, color='blue', linestyle=':', alpha=0.5, label='Interior Target (24.0°C)')

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Thermal Trajectory & Comfort Benchmarking")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # ------------------ Plot 2: CO2 Profile ------------------
    # Baseline
    ax2.plot(base_df['step'], base_df['w_co2'], label='Baseline: Window Zone', 
             color='#ff9999', linestyle='--', marker='o')
    ax2.plot(base_df['step'], base_df['i_co2'], label='Baseline: Interior Zone', 
             color='#9999ff', linestyle='--', marker='^')
    
    # Agentic
    ax2.plot(agent_df['step'], agent_df['w_co2'], label='Agentic: Window Zone', 
             color='#cc0000', linestyle='-', linewidth=2, marker='o')
    ax2.plot(agent_df['step'], agent_df['i_co2'], label='Agentic: Interior Zone', 
             color='#0000cc', linestyle='-', linewidth=2, marker='^')
    
    # Hazardous limit threshold
    ax2.axhline(y=1000, color='purple', linestyle='-.', alpha=0.6, label='Fresh Air Limit (1000 ppm)')

    ax2.set_xlabel("Control Steps (Time Intervals)")
    ax2.set_ylabel("CO2 Concentration (ppm)")
    ax2.set_title("Indoor Air Quality (CO2) Mitigation Dynamics")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Layout adjustments
    plt.tight_layout()
    
    # Save as high-quality PNG and vector SVG for the LaTeX paper
    plt.savefig("hvac_benchmark_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("hvac_benchmark_comparison.svg", format='svg', bbox_inches='tight')
    
    print("\n📈 Visualizations successfully generated!")
    print("   👉 Saved high-resolution PNG: 'hvac_benchmark_comparison.png'")
    print("   👉 Saved vector graphics SVG:  'hvac_benchmark_comparison.svg' (Best for LaTeX papers)")
    
    plt.show()

if __name__ == "__main__":
    generate_publication_plots()
