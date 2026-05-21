import torch
import torch.nn as nn

class RotationalLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.01):
        """
        Rotational Linear Layer (RML) Prototype.
        :param in_features: Size of each input sample.
        :param out_features: Size of each output sample.
        :param alpha: Angular steering coefficient (Rotational Step Size).
        """
        super(RotationalLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Initialize structural weights weights (Must be square for pure matrix_exp rotation, 
        # or properly projected. For this prototype, we assume square feature mapping for simplicity).
        assert in_features == out_features, "For pure Lie-Algebra rotation projection, features must be square in this baseline."
        
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """
        Standard forward pass computing linear mapping.
        """
        return torch.matmul(x, self.weights.t()) + self.bias

    @torch.no_grad()
    def apply_rotational_update(self):
        """
        The Core Innovation:
        Transforms raw Euclidean gradients into a Skew-Symmetric matrix,
        computes the Matrix Exponential to fetch the Orthogonal Rotation Matrix,
        and rotates the weights safely along the geodesic path.
        """
        if self.weights.grad is None:
            return

        G = self.weights.grad.data
        W = self.weights.data

        # 1. Project gradient into Skew-Symmetric Space: Omega^T = -Omega
        # Omega = G . W^T - W . G^T
        omega = torch.matmul(G, W.t()) - torch.matmul(W, G.t())

        # 2. Compute the Matrix Exponential to generate the clean Orthogonal Rotation Matrix (R)
        # R = exp(alpha * omega) -> where R^T . R = I
        rotation_matrix = torch.matrix_exp(self.alpha * omega)

        # 3. Apply the Geodesic Rotational Update to the weights
        # W_new = R . W_old
        updated_weights = torch.matmul(rotation_matrix, W)
        
        # Overwrite parameter data safely
        self.weights.copy_(updated_weights)

# --- Verification & Simulation Loop ---
if __name__ == "__main__":
    print("="*60)
    print("Initializing Geodesic Rotational Update Verification")
    print("="*60)

    # Instantiate our innovative layer (5 features in, 5 features out)
    layer = RotationalLinearLayer(in_features=5, out_features=5, alpha=0.1)
    
    # Simulate synthetic inputs and target losses
    inputs = torch.randn(2, 5)
    targets = torch.randn(2, 5)
    criterion = nn.MSELoss()

    # Track structural matrix properties before update
    initial_norm = torch.norm(layer.weights).item()
    print(f"Initial Weight Frobenius Norm: {initial_norm:.6f}")

    # Execution Loop Iteration
    outputs = layer(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    print(f"Loss Value: {loss.item():.6f}")
    print("Applying Non-Euclidean Rotational update...")
    
    # Execute our newly invented update protocol
    layer.apply_rotational_update()

    # Track structural matrix properties AFTER update
    post_update_norm = torch.norm(layer.weights).item()
    print(f"Post-Update Weight Frobenius Norm: {post_update_norm:.6f}")
    print(f"Norm Variance (Should be near zero / perfectly preserved): {abs(initial_norm - post_update_norm):.6f}")
    print("="*60)
