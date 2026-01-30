import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

def energy_function(x, n):
    """
    Potential Energy Function.
    x is a shape (3, n) array.
    """
    # Use pdist for efficient pairwise distance calculation
    # x.T is (n, 3)
    dists = pdist(x.T)
    
    # Pairwise repulsion sum 1/d using efficient numpy operations
    # Avoid division by zero if any, though random init makes it unlikely
    with np.errstate(divide='ignore'):
        inv_dists = 1.0 / dists
    
    interaction_term = np.sum(inv_dists)
    
    # Harmonic trap |xi|^2
    trap_term = np.sum(np.linalg.norm(x, axis=0)**2)
    
    return interaction_term + trap_term

def gradient_analytical(x, n):
    """
    Analytical gradient of F(x) = Sum(1/|xi-xj|) + Sum(|xi|^2)
    Returns g of shape (3, n).
    """
    g = np.zeros_like(x)
    
    # Gradient of trap term: d/dxi |xi|^2 = 2*xi
    g += 2 * x
    
    # Gradient of repulsion term:
    # d/dxi 1/|xi-xj| = - (xi-xj) / |xi-xj|^3
    
    # Vectorized calculation
    # X_exp: (3, n, 1), X_T_exp: (3, 1, n)
    X_exp = x[:, :, np.newaxis] 
    X_T_exp = x[:, np.newaxis, :] 
    
    diff = X_exp - X_T_exp # (3, n, n) matrix of vectors pointing from j to i
    
    # Distances
    dists = np.linalg.norm(diff, axis=0) # (n, n)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists_cubed = 1.0 / (dists ** 3)
    
    # Remove diagonal (self-interaction) which is Inf
    np.fill_diagonal(inv_dists_cubed, 0.0)
    
    # The gradient contribution for i is sum_j ( - (xi-xj)/|xi-xj|^3 )
    # diff[:, i, j] is xi - xj
    # We multiply diff by inv_dists_cubed. 
    # Broadcasting: (3, n, n) * (n, n) -> (3, n, n)
    
    grad_contributions = - diff * inv_dists_cubed[np.newaxis, :, :]
    
    # Sum over j (axis 2) to get total force on i
    g += np.sum(grad_contributions, axis=2)
    
    return g

def main():
    np.random.seed(42)
    n = 50
    # Initialize random points
    x = np.random.randn(3, n)
    
    T = 2000
    # Learning rate
    # Note: Fixed step size can be unstable for 1/r potential. 
    # Adaptive or small step is better.
    lr = 0.005 
    
    print(f"Starting Gradient Descent for n={n}...")
    
    f_values = []
    
    for t in range(T):
        g = gradient_analytical(x, n)
        
        # Simple Gradient Descent
        x = x - lr * g
        
        if t % 100 == 0:
            f_val = energy_function(x, n)
            f_values.append(f_val)
            print(f"Iteration {t}: Energy = {f_val:.4f}")

    # --- Post Processing ---
    # Calculate effective radii
    dists = pdist(x.T)
    if len(dists) == 0:
        min_dist = 0
    else:
        min_dist = np.min(dists)
        
    r = min_dist / 2.0
    
    dist_from_origin = np.linalg.norm(x, axis=0)
    max_dist_origin = np.max(dist_from_origin)
    
    # Scale to unit spheres (r=1)
    if r > 1e-9:
        scale_factor = 1.0 / r
    else:
        scale_factor = 1.0
        
    R_scaled = max_dist_origin * scale_factor + 1.0 
    
    print("-" * 30)
    print(f"Optimization Result:")
    print(f"Minimum pairwise distance (raw): {min_dist:.6f}")
    print(f"Ratio R/r (Objective): {R_scaled:.6f}")
    print("-" * 30)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scaled_positions = x * scale_factor
    
    ax.scatter(scaled_positions[0, :], scaled_positions[1, :], scaled_positions[2, :], 
               c='b', s=50, depthshade=True, label='Sphere Centers')
    
    # Plot Container
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xc = R_scaled * np.outer(np.cos(u), np.sin(v))
    yc = R_scaled * np.outer(np.sin(u), np.sin(v))
    zc = R_scaled * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(xc, yc, zc, color='r', alpha=0.1)
    
    ax.set_title(f"Gradient Descent Packing (n={n}, R={R_scaled:.4f})")
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()
