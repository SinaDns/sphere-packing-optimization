import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def F(x):
    """
    Potential Energy Function.
    x is a shape (3, n) array.
    """
    n = x.shape[1]
    f = 0.0
    # Pairwise repulsion 1/|xi - xj|
    for i in range(n - 1):
        diff = x[:, i+1:] - x[:, i:i+1] # Shape (3, n-1-i)
        dists = np.linalg.norm(diff, axis=0) # Shape (n-1-i,)
        f += np.sum(1.0 / dists)
    
    # Harmonic trap |xi|^2
    f += np.sum(np.linalg.norm(x, axis=0)**2)
    return f

def grad(x):
    """
    Analytical gradient of F(x).
    Returns g of shape (3, n).
    """
    n = x.shape[1]
    g = np.zeros_like(x)
    
    for i in range(n):
        # Contribution from harmonic trap: 2 * xi
        gi = 2 * x[:, i]
        
        # Contribution from repulsion
        for j in range(n):
            if i != j:
                diff = x[:, i] - x[:, j]
                norm_d = np.linalg.norm(diff)
                if norm_d > 1e-9:
                    gi -= diff / (norm_d**3)
        g[:, i] = gi
    return g

def main():
    np.random.seed(42)
    n = 50
    x0 = np.random.randn(3, n)
    x = x0.copy()
    
    T = 1000
    fvalues = np.zeros(T)
    
    # Gradient Descent with fixed step (from MATLAB code: alpha = 0.2)
    # The MATLAB loop:
    # for t = 1:T
    #   g = reshape(grad(x), 3*n, 1);
    #   p = -g;
    #   alpha = 0.2;
    #   x = x + alpha * reshape(p, 3, n);
    
    lr = 0.2
    
    print("Starting Gradient Descent...")
    for t in range(T):
        g = grad(x)
        x = x - lr * g
        
        f_val = F(x)
        fvalues[t] = f_val
        if t % 100 == 0:
            print(f"Iteration {t}: F(x) = {f_val:.4f}")

    # Calculate Radius
    min_distance = np.inf
    max_distance_from_center = 0.0
    
    for i in range(n):
        # Min pairwise distance implies diameter of small spheres?
        # In MATLAB: r = min_distance / 2
        for j in range(i + 1, n):
            d = np.linalg.norm(x[:, i] - x[:, j])
            if d < min_distance:
                min_distance = d
        
        d_center = np.linalg.norm(x[:, i])
        if d_center > max_distance_from_center:
            max_distance_from_center = d_center

    r = min_distance / 2.0
    R = max_distance_from_center + r
    
    print(f"Small sphere radius (r): {r:.12f}")
    print(f"Container radius/r ratio (R/r): {R/r:.12f} (Target approx optimization metric)")

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw spheres
    # We can't easily draw true 3d spheres in matplotlib without surface plots for each
    # Approximating with valid scatter or drawing wireframes
    
    # Draw container
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    container_x = R * np.outer(np.cos(u), np.sin(v))
    container_y = R * np.outer(np.sin(u), np.sin(v))
    container_z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(container_x, container_y, container_z, color='c', alpha=0.1)
    
    # Draw small spheres (points for simplicity, or small spheres)
    # Using scatter with size correlated to radius is tricky in 3d.
    # We'll just scatter the centers
    ax.scatter(x[0, :], x[1, :], x[2, :], c='b', s=20, label='Sphere Centers')
    
    # Optional: Draw actual small spheres for a few or all if fast
    # (Skipping dense mesh for 50 spheres to keep performance sane)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sphere Packing (Gradient Descent)')
    plt.show()

if __name__ == "__main__":
    main()
