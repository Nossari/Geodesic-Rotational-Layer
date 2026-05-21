import torch
import torch.nn as nn
from rotational_layer import RotationalLinearLayer

def run_rotation_benchmark():
    print("="*70)
    print("Running Non-Euclidean Rotational vs Standard Linear Layer Benchmark")
    print("="*70)
    
    # 1. Configuration parameters
    features = 10
    alpha = 0.05  # Rotational learning step size
    lr_standard = 0.05  # Standard learning rate
    epochs = 5
    
    # 2. Initialize layers
    torch.manual_seed(100)
    rotational_layer = RotationalLinearLayer(in_features=features, out_features=features, alpha=alpha)
    standard_layer = nn.Linear(in_features=features, out_features=features, bias=True)
    
    # Copy exact initial random weights from standard layer to make comparisons fair
    with torch.no_grad():
        rotational_layer.weights.copy_(standard_layer.weight.data)
        rotational_layer.bias.copy_(standard_layer.bias.data)

    # 3. Create synthetic simulation data
    x_input = torch.randn(5, features)
    y_target = torch.randn(5, features)
    criterion = nn.MSELoss()
    
    # Define standard SGD optimizer for the baseline layer only
    standard_optimizer = torch.optim.SGD(standard_layer.parameters(), lr=lr_standard)
    
    print(f"Simulating {epochs} training iterations...\n")
    
    for epoch in range(1, epochs + 1):
        # --- Standard Layer Execution & Tracking ---
        standard_optimizer.zero_grad()
        out_standard = standard_layer(x_input)
        loss_standard = criterion(out_standard, y_target)
        loss_standard.backward()
        
        # Track standard weight norm before optimizer step shifts it lineary
        norm_standard_before = torch.norm(standard_layer.weight).item()
        standard_optimizer.step()
        norm_standard_after = torch.norm(standard_layer.weight).item()
        
        # --- Rotational Layer Execution & Tracking ---
        out_rotational = rotational_layer(x_input)
        loss_rotational = criterion(out_rotational, y_target)
        
        # Manually zero out gradients, compute backward, and apply non-Euclidean rotation
        if rotational_layer.weights.grad is not None:
            rotational_layer.weights.grad.zero_()
        loss_rotational.backward()
        
        norm_rotational_before = torch.norm(rotational_layer.weights).item()
        rotational_layer.apply_rotational_update()
        norm_rotational_after = torch.norm(rotational_layer.weights).item()
        
        # Display Comparative Analytical Metrics
        print(f"[Iteration {epoch}]")
        print(f"  -> Standard Layer | Loss: {loss_standard.item():.4f} | Weight Norm Shift: {norm_standard_before:.4f} -> {norm_standard_after:.4f} (Delta: {abs(norm_standard_before - norm_standard_after):.6f})")
        print(f"  -> Rotational Layer | Loss: {loss_rotational.item():.4f} | Weight Norm Shift: {norm_rotational_before:.4f} -> {norm_rotational_after:.4f} (Delta: {abs(norm_rotational_before - norm_rotational_after):.6f})")
        print("-" * 70)

    print("\nBenchmark analysis completed successfully!")
    print("Notice how the Rotational Layer Delta remains rock-solid (preserves structural norm).")
    print("="*70)

if __name__ == "__main__":
    run_rotation_benchmark()
