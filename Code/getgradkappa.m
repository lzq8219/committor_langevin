function [kappa, gradkappa] = getgradkappa(a, sigma, ...
    xmin, xmax, ymin, ymax, nx, ny, ...
    h, l, scale, ...
    signal)

%% Documentation. 
% given parameters a, we calculate the gradient of kappa(a) with 
% respect to a.

% signal = 1 : only return kappa
% signal = 2 :  return both 

% This function is to solve 2d tpt problem with 
%   \sqrt{D} = \sigma I_{2\times 2}.
% The equation is \dot{X}_t = b(X_t)dt+\sqrt{2D} dW_t.
% a = sigma^2 I.


addpath(genpath(pwd));

%% parameters
epsilon = sigma^2;

hx = (xmax-xmin)/nx;
hy = (ymax-ymin)/ny;
x_whole = linspace(xmin, xmax, nx+1); % the whole space
y_whole = linspace(ymin, ymax, ny+1);

x_s = x_whole(2:nx); %truncated x (shorted).
y_s = y_whole(2:ny); %truncated y (shorted).
nx_s = nx-1; %shorted nx.
ny_s = ny-1; %shorted ny.

option = 0.4; %parameter to control the range of set A and B.

% parameters for the perturbation part.


% [X, Y] = meshgrid(x_s, y_s);
% Y = flipud(Y);


b1_fun = @(x, y) b1(x, y, a, h, l, scale);
b2_fun = @(x, y) b2(x, y, a, h, l, scale);

%% solve invariant pdf
rho = inv_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma, b1_fun, b2_fun);


%% plot bolzmann distribution
% bolz = zeros(size(X));
% pote = zeros(size(X));
% for i = 1:size(bolz, 1)
%     for j = 1:size(bolz, 2)
%         pote(i,j) = potential_V(X(i,j), Y(i,j), a, h, l, scale);
%         bolz(i,j) = exp(-pote(i,j)/sigma^2);
%     end
% end
% Z = sum(sum(bolz))*hx*hy;   % the boundary point has 
% 							% very small contribution to Z, 
%                             % so, I can ignore.
% figure, contour(X,Y,bolz/Z,20),colorbar;
% % figure, contour(X, Y, pote, 20), colorbar;



%% solve committor functions
q = back_kol_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma,...
    b1_fun, b2_fun, @(x)in_a(x, option), @(x)in_b(x,option), 0, 1, 0);
% contourtpt(q, nx_s, ny_s, X, Y, sigma);

m = 1 - hx*hy*sum(rho.*q);
[q1, q2] = gradient_q(q, nx_s, ny_s, hx, hy); 
nu = epsilon*hx*hy*sum(rho.*(q1.^2+q2.^2));
kappa = nu/m;


%% get grad U w.r.t. a.
if signal == 2 % return both kappa, and gradkappa.
    % solve w.
    w = back_kol_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma, ...
        b1_fun, b2_fun, @(x) in_a(x, option), @(x) in_b(x, option), ...
        0, 0, -1);
    [w1, w2] = gradient_q(w, nx_s, ny_s, hx, hy);

    dk_du = rho/m.*(...
        kappa*(1-q)/epsilon ...
        + kappa*(q1.*w1+q2.*w2) ...
        - (q1.^2+q2.^2));

    gradu_to_a_result = zeros(length(m), length(a));
    gradu_to_a_fun = @(x, y) gradu_to_a(x, y, a, h, l, scale);
    for i = 1:nx_s
        for j = 1:ny_s    
            idx = (j-1)*nx_s+i;
            gradu_to_a_result(idx,:) = gradu_to_a_fun(x_s(i), y_s(j)); 
        end
    end

    gradkappa = zeros(size(a));
    for i = 1:length(a)
       gradkappa(i) = hx*hy*sum(dk_du.*gradu_to_a_result(:,i));
    end
else
    gradkappa = [];
end

end