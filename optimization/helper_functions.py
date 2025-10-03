import numpy as np


def print_model_evaluation_table(zone_name: str, temp_metrics: dict, power_metrics: dict):
    """
    Print model evaluation metrics in a combined formatted table
    
    Args:
        zone_name: Name of the zone
        temp_metrics: Dictionary containing temperature model MAE, MAPE, R2 values
        power_metrics: Dictionary containing power model MAE, MAPE, R2 values
    """
    print(f"\n{'='*70}")
    print(f"📊 Model Evaluation - {zone_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<15} | {'Temperature':<20} | {'Power':<20}")
    print(f"{'-'*15}-+-{'-'*20}-+-{'-'*20}")
    
    # Format and print each metric
    for metric_name in ["MAE", "MAPE", "R2"]:
        if metric_name == "MAE":
            temp_val = f"{temp_metrics[metric_name]:.2f}°C"
            power_val = f"{power_metrics[metric_name]:.3f} kW"
        elif metric_name == "MAPE":
            temp_val = f"{temp_metrics[metric_name]:.1f}%"
            power_val = f"{power_metrics[metric_name]:.1f}%" if not np.isnan(power_metrics[metric_name]) else "N/A"
        else:  # R2
            temp_val = f"{temp_metrics[metric_name]:.3f}"
            power_val = f"{power_metrics[metric_name]:.3f}"
        
        print(f"{metric_name:<15} | {temp_val:<20} | {power_val:<20}")
    
    print(f"{'='*70}")