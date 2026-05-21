# Geodesic Rotational Layer (GRL) for Non-Euclidean Optimization

This repository introduces an innovative structural approach to deep learning parameter optimization. Instead of standard Euclidean translational updates ($W \leftarrow W - \eta G$), this architecture projects first-order gradients into a Skew-Symmetric space, transforming standard dense weights through clean **Matrix Exponentials** to enforce continuous rotational trajectories on a curved manifold.

---

## 🔬 Mathematical Formulation

Standard optimization paths trigger feature decay or exploding parameter behavior because matrix modifications are strictly linear and cumulative. The **Geodesic Rotational Layer** enforces constant norm trajectories using Lie-Algebraic properties.

### 1. Skew-Symmetric Projection
Given weight matrix $\mathbf{W}$ and its computed gradient $\mathbf{G}$, we extract the inner asymmetric tensor structure to create an empirical manifold tangent vector:
$$\mathbf{\Omega} = \mathbf{G}\mathbf{W}^T - \mathbf{W}\mathbf{G}^T$$
By definition, this satisfies the perfect rotational condition: $\mathbf{\Omega}^T = -\mathbf{\Omega}$.

### 2. Matrix Exponential Map & Parameter Update
We project the geometric rotation safely into the compact Lie group $SO(n)$ using the matrix exponential:
$$\mathbf{R} = \exp(\alpha \cdot \mathbf{\Omega})$$
The weight updates are then driven via strict matrix multiplication:
$$\mathbf{W}_{new} = \mathbf{R} \cdot \mathbf{W}_{old}$$

Because $\mathbf{R}^T\mathbf{R} = \mathbf{I}$, the matrix Frobenius norm is perfectly preserved across arbitrary training epochs, eradicating the mathematical requirement for auxiliary structural constraints like `Weight Decay`.

---

## 🚀 Key Advantages
* **Strict Norm-Preservation:** Completely immune to exploding/vanishing parameters by design.
* **Geodesic Trajectories:** Moves parameters along the absolute shortest curved geometric path toward convergence.
* **Memory and Compute Efficient:** Operates perfectly inside standard eager execution configurations via `torch.matrix_exp`.

---

## 📜 License
This project is open-sourced under the terms of the **MIT License**.
