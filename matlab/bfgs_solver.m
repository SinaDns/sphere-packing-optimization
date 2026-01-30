clear;
clc;

n = 100;
R = nthroot(n / 0.6, 3);
positions = rand(n, 3) * 2 * (R - 1) - (R - 1);

max_iterations = 1000; 
tolerance = 1e-6; 

H = eye(n * 3); 
grad_old = gradient(positions, R); 
positions_old = positions;

for iter = 1:max_iterations

    direction = -H * grad_old(:);
    direction = reshape(direction, n, 3);

    alpha = line_search(positions, direction, R);

    positions = positions + alpha * direction;

    grad_new = gradient(positions, R);

    s = positions(:) - positions_old(:);
    y = grad_new(:) - grad_old(:);

    rho = 1 / (y' * s);
    H = (eye(n * 3) - rho * s * y') * H * (eye(n * 3) - rho * y * s') + rho * s * s';

    if norm(positions - positions_old, 'fro') < tolerance
        fprintf('Converged after %d iterations\n', iter);
        break;
    end

    positions_old = positions;
    grad_old = grad_new;
end

positions = reshape(positions, n, 3);

[positions, R_star] = adjust_container(positions, R);

figure;
hold on;

[x, y, z] = sphere;
surf(R_star * x, R_star * y, R_star * z, 'FaceAlpha', 0.1, 'EdgeColor', 'none');

colors = rand(n, 3);

for i = 1:n
    pos = positions(i, :);
    surf(pos(1) + 1*x, pos(2) + 1*y, pos(3) + 1*z, 'FaceColor', colors(i, :), 'FaceAlpha', 0.9, 'EdgeColor', 'none');
end

axis equal;
title('Packing of Unit Spheres in a Larger Sphere');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
grid on;
view(3);

fprintf('R_star: %.12f\n', R_star);


function [new_positions, R_star] = adjust_container(positions, R)
    lambda = 1e-4;
    new_positions = positions;

    for i = 1:35
        max_dist = max(sqrt(sum(new_positions.^2, 2))) + 1;
        if max_dist > R
            R = max_dist;
            new_positions = positions;
        end
        lambda = 0.5 * lambda;
    end
    R_star = R;
end


function grad = gradient(positions, R)
    n = size(positions, 1);
    grad = zeros(size(positions));

    h = 1e-6;

    for i = 1:n
        for k = 1:3
            positions_h = positions;
            positions_h(i, k) = positions_h(i, k) + h;
            E_plus = energy(positions_h, R);

            positions_h(i, k) = positions(i, k) - h;
            E_minus = energy(positions_h, R);

            grad(i, k) = (E_plus - E_minus) / (2 * h);
        end
    end
end


function E = energy(positions, R)
    n = size(positions, 1);
    E = 0;

    for i = 1:n
        for j = i + 1:n
            dist = norm(positions(i, :) - positions(j, :));
            if dist < 2
                E = E + (2 - dist)^2;
            end
        end
        if norm(positions(i, :)) > R - 1
            E = E + (norm(positions(i, :)) - (R - 1))^2;
        end
    end
end


function alpha = line_search(positions, direction, R)
    alpha = 1; 
    rho = 0.5; 
    c = 0.01; 

    E_initial = energy(positions, R);
    grad = gradient(positions, R); 

    while energy(positions + alpha * direction, R) > E_initial + c * alpha * sum(grad(:) .* direction(:))
        alpha = rho * alpha;
        if alpha < 1e-10
            break; 
        end
    end
end

