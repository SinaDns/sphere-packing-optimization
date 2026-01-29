import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def energy(positions, R):
    """
    Returns energy E based on overlaps and boundary constraint.
    positions: (n, 3)
    R: float container radius
    """
    n = positions.shape[0]
    E = 0.0
    
    # Pairwise overlap penalty (distance < 2 means overlap for unit spheres of radius 1)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2.0:
                E += (2.0 - dist)**2
    
    # Boundary penalty
    # If center is > R - 1, it's poking out (center must be within R-1)
    norms = np.linalg.norm(positions, axis=1)
    # Mask for items outside boundary
    mask = norms > (R - 1.0)
    # Add penalty (dist_from_center - (R-1))^2
    E += np.sum((norms[mask] - (R - 1.0))**2)
    
    return E

def gradient_finite_diff(positions, R, h=1e-6):
    """
    Numerical gradient using central differences.
    """
    n, d = positions.shape
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for k in range(d):
            # Save original
            orig = positions[i, k]
            
            # Plus
            positions[i, k] = orig + h
            E_plus = energy(positions, R)
            
            # Minus
            positions[i, k] = orig - h
            E_minus = energy(positions, R)
            
            # Restore
            positions[i, k] = orig
            
            grad[i, k] = (E_plus - E_minus) / (2 * h)
            
    return grad

def line_search(positions, direction, R):
    alpha = 1.0
    rho = 0.5
    c = 0.01
    
    E_initial = energy(positions, R)
    grad_val = gradient_finite_diff(positions, R)
    
    # projected gradient along direction: sum(g * d)
    desc_val = np.sum(grad_val * direction)
    
    while True:
        pos_new = positions + alpha * direction
        E_new = energy(pos_new, R)
        if E_new <= E_initial + c * alpha * desc_val:
            break
        alpha *= rho
        if alpha < 1e-10:
            break
            
    return alpha

def adjust_container(positions, R):
    """
    Iteratively shrinks radius R if possible? 
    MATLAB logic:
    for i = 1:35
        max_dist = max(sqrt(sum(new_positions.^2, 2))) + 1;
        if max_dist > R  <-- This looks like it GROWS R if points are outside?
            R = max_dist;
            new_positions = positions; <-- Resets positions?
        end
        lambda = 0.5 * lambda; <-- lamba used? Matlab code defined lambda=1e-4 but didn't use it except to multiply it.
        Wait, the MATLAB code for adjust_container is weird.
        It defines lambda, updates specific things..
        Actually, looking at MATLAB BFGS 'adjust_container':
        It just ensures R contains all spheres (R = max_dist) if they are outside.
        It doesn't seem to actively *shrink* R in the loop provided, just validation.
    """
    # Simply calculate the minimal bounding sphere radius centered at 0 that contains all unit spheres
    # Max distance of center from origin + 1 (unit radius)
    max_dist = np.max(np.linalg.norm(positions, axis=1)) + 1.0
    return positions, max_dist

def main():
    np.random.seed(0)
    n = 100
    # R initialization: sphere volume V = 4/3 pi R^3. n spheres volume = n * 4/3 pi 1^3.
    # Packing density ~ 0.6.  n * 4/3 pi / (4/3 pi R^3) = 0.6 => n / R^3 = 0.6 => R = (n/0.6)^(1/3)
    R = (n / 0.6)**(1/3)
    
    positions = np.random.rand(n, 3) * 2 * (R - 1) - (R - 1)
    
    max_iterations = 1000
    tolerance = 1e-6
    
    H = np.eye(n * 3)
    grad_old = gradient_finite_diff(positions, R)
    positions_old = positions.copy()
    
    print(f"Starting BFGS for {n} spheres in R={R:.4f}...")
    
    for iter_num in range(max_iterations):
        grad_flat = grad_old.flatten()
        
        # direction = -H * grad
        direction_flat = -H @ grad_flat
        direction = direction_flat.reshape(n, 3)
        
        alpha = line_search(positions, direction, R)
        
        positions = positions + alpha * direction
        
        grad_new = gradient_finite_diff(positions, R)
        
        # BFGS Update
        s = (positions - positions_old).flatten()
        y = (grad_new - grad_old).flatten()
        
        if np.dot(y, s) > 1e-10: # Avoid division by zero
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n * 3)
            # H = (I - rho * s * y.T) * H * (I - rho * y * s.T) + rho * s * s.T
            # Using outer products
            # This is O((3n)^2), for n=100 -> 300x300 matrix, it's fine.
            
            term1 = I - rho * np.outer(s, y)
            term2 = I - rho * np.outer(y, s)
            H = term1 @ H @ term2 + rho * np.outer(s, s)
        
        if np.linalg.norm(positions - positions_old) < tolerance:
            print(f'Converged after {iter_num} iterations')
            break
            
        positions_old = positions.copy()
        grad_old = grad_new.copy()
        
        if iter_num % 10 == 0:
            print(f"Iteration {iter_num}, Energy: {energy(positions, R):.4f}")

    positions, R_star = adjust_container(positions, R)
    print(f"Final Container Radius R_star: {R_star:.12f}")
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    # Draw container
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    container_x = R_star * np.outer(np.cos(u), np.sin(v))
    container_y = R_star * np.outer(np.sin(u), np.sin(v))
    container_z = R_star * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(container_x, container_y, container_z, color='gray', alpha=0.3)
    
    # Draw spheres
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', s=50) # s is area usually
    
    ax.set_title('BFGS Sphere Packing')
    plt.show()

if __name__ == "__main__":
    main()
