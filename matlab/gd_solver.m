clear;  
clc;  
format long;

n = 50;  
x0 = randn(3,n);  
x = x0;
%% 



T = 1000;  
fvalues = zeros(1,T);  
gvalues = zeros(1,T);  
alphavalues = zeros(1,T);  

for t = 1:T  
    g = reshape(grad(x),3*n,1);
    p = -g;
    %B = Hessian(x);
    %p = B\(-g); 
    
    %alpha = backtrack(x,reshape(g,3,n),reshape(p,3,n));  
    alpha = 0.2;
    x = x + alpha*reshape(p,3,n);  

    fvalues(t) = F(x);  
    gvalues(t) = norm(g);  
    alphavalues(t) = alpha;  

    plot3(x(1,:),x(2,:),x(3,:),'.','markersize',15);  
    axis equal; 
    title(['Iteration: ', num2str(t)]);   
    grid on;   
    drawnow;  
end
%B = Hessian(x);

min_distance = inf; 
max_distance_from_center = 0;

for i = 1:n  
    for j = i+1:n
        distance = norm(x(:,i) - x(:,j));  
        if distance < min_distance  
            min_distance = distance;
        end  
    end  
    
    distance_from_center = norm(x(:,i));  
    if distance_from_center > max_distance_from_center  
        max_distance_from_center = distance_from_center;
    end
end  


r = min_distance/2;
R = max_distance_from_center + r;
fprintf('min: %.12f\n', r);  
fprintf('max: %.12f\n', R/r);

[SphereX, SphereY, SphereZ] = sphere(20);

hold on;
for i = 1:n  
    surf(SphereX * r + x(1,i), SphereY * r + x(2,i), SphereZ * r + x(3,i), 'FaceColor', 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'none');  
end 

[x, y, z] = sphere;   
surf(R * x, R * y, R * z, 'FaceAlpha', 0.1, 'EdgeColor', 'none');

hold off;
title('Points with Spheres of Radius r');  
xlabel('X-axis');  
ylabel('Y-axis');  
zlabel('Z-axis');  
axis equal;
grid on;  


function f = F(x)  
    n = size(x, 2);  
    f = 0;  
    for i = 1:(n-1)  
        for j = (i+1):n  
            %f = f - log(norm(x(:,i)-x(:,j)));  
            f = f + 1/norm(x(:,i)-x(:,j));        
        end  
    end  
    for i = 1:n  
        f = f + (norm(x(:,i))^2);
    end  
end  


function g = grad(x)  
    n = size(x, 2);  
    g = zeros(3,n);  
    for i = 1:n  
        gi = 2 * x(:,i);  
        for j = 1:n  
            if j ~= i  
                gi = gi - (x(:,i)-x(:,j))/norm(x(:,i)-x(:,j))^3;  
            end  
        end  
        g(:,i) = gi;   
    end  
end


function alpha = backtrack(x,g,p)  
    alpha = 1;  
    rho = 0.9;  
    c = 0.01;  
    while (F(x+alpha*p) > F(x) + c * alpha * sum(sum(g.*p)))  
        alpha = alpha * rho;  
    end  
end  


function H = Hessian(x)  
    n = size(x, 2);  
    H = zeros(3*n);
    
    for i = 1:n  
        for j = 1:n  
            if j ~= i         
                d = x(:, i) - x(:, j); 
                norm_d = norm(d);  
                K = (1/norm_d^3) * (eye(3) - (d * d') / norm_d^2); 
                H(3*i-2:3*i, 3*j-2:3*j) = H(3*i-2:3*i, 3*j-2:3*j) + K; 
                H(3*j-2:3*j, 3*i-2:3*i) = H(3*j-2:3*j, 3*i-2:3*i) + K; 
            end  
        end  
    end  

    for i = 1:n  
        K = 2 * eye(3);
        H(3*i-2:3*i, 3*i-2:3*i) = H(3*i-2:3*i, 3*i-2:3*i) + K;  
    end  
end