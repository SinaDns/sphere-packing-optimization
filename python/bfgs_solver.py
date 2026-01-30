import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BFGSSolver:
    def __init__(self, n_spheres, tol=1e-5, max_iter=1000):
        self.n = n_spheres
        self.tol = tol
        self.max_iter = max_iter
        
        # Initial guess for R (packing density ~ 0.6)
        # Volume of n unit spheres = n * 4/3 * pi
        # Volume of container R = 4/3 * pi * R^3
        # n / R^3 ~ 0.6  => R ~ (n/0.6)^(1/3)
        self.R_initial = (self.n / 0.6)**(1.3) # Slightly larger start
        
    def energy(self, x, R):
        """
        Energy function to minimize (Penalty Method).
        x: positions (n, 3) flattened to (3n,)
        R: Container radius
        """
        positions = x.reshape((self.n, 3))
        E = 0.0
        
        # 1. Overlap Penalty
        # Efficient pairwise distance calculation
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        # dists[i,j] is distance between i and j
        dists = np.linalg.norm(diff, axis=2)
        
        # We only care about i < j to avoid double counting and self-distance
        # Overlap occurs if dist < 2 (since radius=1)
        # Using upper triangular indices
        iu, ju = np.triu_indices(self.n, k=1)
        
        pair_dists = dists[iu, ju]
        # Penalty function: (2 - d)^2 if d < 2 else 0
        overlap_mask = pair_dists < 2.0
        if np.any(overlap_mask):
             E += np.sum((2.0 - pair_dists[overlap_mask])**2)
             
        # 2. Boundary Constraint
        # Each sphere center must be within R-1 from origin
        dist_from_origin = np.linalg.norm(positions, axis=1)
        # Violation if dist > R - 1
        boundary_limit = R - 1.0
        # If R < 1, then boundary_limit is negative, meaning physically impossible even for 1 sphere
        # But we handle math:
        
        violation_mask = dist_from_origin > boundary_limit
        if np.any(violation_mask):
            # Penalty: (dist - (R-1))^2
            E += np.sum((dist_from_origin[violation_mask] - boundary_limit)**2)
            
        return E

    def gradient(self, x, R):
        """
        Analytical Gradient of Energy with respect to positions x.
        """
        positions = x.reshape((self.n, 3))
        grad = np.zeros_like(positions)
        
        # 1. Overlap Gradient
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :] # (N, N, 3)
        dists = np.linalg.norm(diff, axis=2) # (N, N)
        
        # Avoid division by zero on diagonal elements
        with np.errstate(divide='ignore', invalid='ignore'):
             inv_dists = 1.0 / dists
        inv_dists[np.diag_indices(self.n)] = 0.0
        
        # Mask for overlaps: dist < 2 and i != j
        # We process full matrix for vectorization (forces are symmetric)
        # Force magnitude derivation: E = (2-d)^2 => dE/dPos = 2(2-d)(-1) * (diff/d) = -2(2-d)/d * diff
        # Actually dE/dd = 2(2-d)*(-1) = 2(d-2). 
        # d(d)/dx_i = (x_i - x_j)/d
        # So grad_i = 2(d-2) * (x_i - x_j)/d = 2(1 - 2/d) * (x_i - x_j)
        
        overlap_mask = (dists < 2.0) & (dists > 1e-8) # Avoid self
        
        if np.any(overlap_mask):
            # Term: 2 * (d - 2) * (1/d) * (xi - xj)
            #     = 2 * (1 - 2/d) * (xi - xj)
            factor = 2.0 * (1.0 - 2.0 * inv_dists)
            # Apply mask
            factor[~overlap_mask] = 0.0
            
            # Sum over j to get force on i
            # grad[i] += sum_j factor[i,j] * (pos[i] - pos[j])
            # This is equivalent to dot product on axes
            
            # Broadcasting: factor is (N,N), diff is (N,N,3)
            # We want sum over axis 1 (j)
            grad += np.sum(factor[:, :, np.newaxis] * diff, axis=1)

        # 2. Boundary Gradient
        dists_origin = np.linalg.norm(positions, axis=1)
        limit = R - 1.0
        mask = dists_origin > limit
        if np.any(mask):
            # E = (d_org - limit)^2
            # dE/dx = 2(d_org - limit) * x / d_org
            #       = 2(1 - limit/d_org) * x
            
            # Calculate gradient contribution from boundary penalty.
            # The mask ensures we only compute this for spheres violating the boundary.
            
            # Vectorized
            current_pos = positions[mask]
            cur_dists = dists_origin[mask]
            
            factor_b = 2.0 * (1.0 - limit / cur_dists)
            grad[mask] += factor_b[:, np.newaxis] * current_pos
            
        return grad.flatten()

    def minimize_r_fixed(self, R, initial_x):
        """
        Run BFGS for fixed R.
        """
        x = initial_x.copy()
        n_dim = 3 * self.n
        
        # Initial Hessian approximation (Identity)
        B_inv = np.eye(n_dim) # Inverse Hessian
        
        grad = self.gradient(x, R)
        
        for k in range(self.max_iter):
            gnorm = np.linalg.norm(grad)
            if gnorm < self.tol:
                break
                
            # Direction
            p = -B_inv @ grad
            
            # Line Search (Backtracking Armijo)
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            E_old = self.energy(x, R)
            
            while alpha > 1e-10:
                x_new = x + alpha * p
                E_new = self.energy(x_new, R)
                if E_new <= E_old + c1 * alpha * np.dot(grad, p):
                    break
                alpha *= rho
            
            # Update
            s = x_new - x
            y = self.gradient(x_new, R) - grad
            
            # BFGS Update
            # Skip if s or y is too small
            if np.linalg.norm(s) > 1e-14 and np.linalg.norm(y) > 1e-14:
                rho_inv = np.dot(y, s)
                if rho_inv > 1e-10: # Curvature condition
                    rho_k = 1.0 / rho_inv
                    I = np.eye(n_dim)
                    V = I - rho_k * np.outer(s, y)
                    B_inv = V @ B_inv @ V.T + rho_k * np.outer(s, s)
            
            x = x_new
            grad = self.gradient(x, R)  # Calc new grad for next step
            
        return x, self.energy(x, R)

    def solve(self):
        # Initialize random positions within the container.
        # The optimizer will move spheres into valid configurations.
        limit = max(0.1, self.R_initial - 1.0)
        x = (np.random.rand(self.n * 3) - 0.5) * 2 * limit
        
        current_R = self.R_initial
        
        print(f"Starting BFGS Optimization with n={self.n} spheres...")
        
        # Iterative shrinking strategy:
        # 1. Minimize energy for a fixed radius R.
        # 2. If successful, shrink R based on the current configuration.
        # 3. If unsuccessful (high energy), expand R to recover.
        
        step_r = 0.95
        best_x = x.copy()
        best_R = current_R
        
        for i in range(20): # Outer iterations for R
            # 1. Minimize Energy for current R
            x, E_val = self.minimize_r_fixed(current_R, x)
            
            # 2. Check validity
            # Calculate max distance from origin + 1
            pos = x.reshape((self.n, 3))
            max_dist = np.max(np.linalg.norm(pos, axis=1)) + 1.0
            
            # Check for overlaps
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2) + np.eye(self.n)*10
            min_sep = np.min(dists)
            
            valid_overlap = min_sep >= 1.99 # Tolerance
            
            print(f"Iter {i}: R_env={current_R:.4f}, MaxReach={max_dist:.4f}, MinSep={min_sep:.4f}, E={E_val:.6f}")
            
            if E_val < 1e-4: # Feasible configuration found
                # Configuration fits. Update best R to the tightest bound.
                # Then attempt to shrink further for the next iteration.
                best_R = max_dist
                best_x = x.copy()
                current_R = max_dist * 0.98 # Aggressive shrink
            else:
                # Configuration invalid (overlaps or boundary violation).
                # Increase R to relax constraints.
                current_R = current_R * 1.05
                
        return best_x, best_R

    def plot_solution(self, x, R):
        positions = x.reshape((self.n, 3))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw container
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs = R * np.cos(u) * np.sin(v)
        ys = R * np.sin(u) * np.sin(v)
        zs = R * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.3)
        
        # Draw spheres
        for i in range(self.n):
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
            xs = np.cos(u) * np.sin(v) + positions[i,0]
            ys = np.sin(u) * np.sin(v) + positions[i,1]
            zs = np.cos(v) + positions[i,2]
            ax.plot_surface(xs, ys, zs, color=np.random.rand(3,), alpha=0.8)
            
        ax.set_aspect('auto') # 'equal' often buggy in mpl 3d
        plt.title(f"BFGS Packing n={self.n}, R={R:.4f}")
        plt.show()

if __name__ == "__main__":
    solver = BFGSSolver(n_spheres=20)
    final_x, final_R = solver.solve()
    print(f"Final Radius found: {final_R:.5f}")
    solver.plot_solution(final_x, final_R)
