%% Documentation. 
% This function is to solve 2d tpt problem with 
%   \sqrt{D} = \sigma I_{2\times 2}.
% The equation is \dot{X}_t = b(X_t)dt+\sqrt{2D} dW_t.
% a = sigma^2 I.

addpath(genpath(pwd));

%% parameters
sigma = 1/0.15;

xmin = -3;
xmax = 3;
ymin = -3;
ymax = 3;

nx = 500;
ny = 500;
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
a = [-0.822, 0.624, 0.212, 0.293];
h = 50;
l = 0.25;
scale = 5e-3;


[X, Y] = meshgrid(x_s, y_s);
Y = flipud(Y);


b1_fun = @(x, y) b1(x, y, a, h, l, scale);
b2_fun = @(x, y) b2(x, y, a, h, l, scale);

b1_fun = @(x, y) 0;
b2_fun = @(x, y) 0;

%% solve m(x).
m = inv_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma, b1_fun, b2_fun);

% plot bolzmann distribution
bolz = zeros(size(X));
pote = zeros(size(X));
for i = 1:size(bolz, 1)
    for j = 1:size(bolz, 2)
        pote(i,j) = potential_V(X(i,j), Y(i,j), a, h, l, scale);
        bolz(i,j) = exp(-pote(i,j)/sigma^2);
    end
end
Z = sum(sum(bolz))*hx*hy;   % the boundary point has 
							% very small contribution to Z, 
                            % so, I can ignore.
% figure, contour(X,Y,bolz/Z,20),colorbar;
%figure, contour(X, Y, pote, 20), colorbar;



%% solve committor functions
% q_plus = committor_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma,...
%     @in_a, @in_b, option);
% q_minus = committor_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma,...
%     @in_b, @in_a, option);
% Alternative subroutine.
q_plus = back_kol_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma,...
    b1_fun, b2_fun, @(x)in_a(x, option), @(x)in_b(x,option), 0, 0, -1);
q_plus_mesh = reshape(q_plus,499,499);
q_plus_mesh = rot90(q_plus_mesh);
figure;contour(X, Y, q_plus_mesh, 20);colorbar;


% q_minus = back_kol_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma,...
%     b1_fun, b2_fun, @(x)in_a(x, option), @(x)in_b(x,option), 1, 0, 0);
% 
% %% find first passage time.
% % % fpt = back_kol_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma,...
% % %     @b1, @b2, @(x)in_a(x, option), @(x)in_b(x, option), 0, 0, -1);
% % fpt1 = back_kol_solve(x_s, y_s, nx_s, ny_s, hx, hy, 0.6,...
% %      @b1, @b2, @(x)in_a(x, option), @(x)in_b(x, option), 0, 0, -1);
% % fpt2 = back_kol_solve(x_s, y_s, nx_s, ny_s, hx, hy, 0.5,...
% %      @b1, @b2, @(x)in_a(x, option), @(x)in_b(x, option), 0, 0, -1);
% % 
% % % visualize data.
% % fpt1_new = rot90(reshape(fpt1,nx_s,ny_s));
% % fpt2_new = rot90(reshape(fpt2,nx_s,ny_s));
% % 
% % f1 = figure();
% % contourf(X,Y,fpt1_new,20),title(['sigma = ',num2str(0.6)]),...
% %     colormap(jet),colorbar;
% % f2 = figure(); 
% % contourf(X,Y,fpt2_new,20),title(['sigma = ',num2str(0.5)]),...
% %     colormap(jet),colorbar;
% % f3 = figure();hold on;
% % largemat = zeros(size(X));
% % smallmat = zeros(size(X));
% % index = find(fpt1_new-fpt2_new > 0);
% % largemat(index) = ones(length(index),1);
% % spy(largemat,'b');
% % index = find(fpt1_new-fpt2_new < 0);
% % smallmat(index) = ones(length(index),1);
% % spy(smallmat,'r');
% 
% 
% %% get m_ab_x.
% m_r_x = q_minus.*m.*q_plus; % m^R(x).
% z_ab = sum(m_r_x)*hx*hy;
% m_ab_x = m_r_x/z_ab;
% M_AB_X = rot90(reshape(m_ab_x, nx_s, ny_s));
% figure, contourf(X, Y, M_AB_X, 15), colorbar;
% 
% %% get Current J.
% % get J. it is not necessary here.
% % [J1,J2] = get_probability_current(x_s, y_s, nx_s, ny_s, ...
% %     hx, hy, sigma, m);
% % 
% J1 = zeros(nx_s*ny_s,1);
% J2 = zeros(nx_s*ny_s,1);
% 
% %% get J^{AB} and rate.
% [J_ab_1, J_ab_2] = get_j_ab(m, q_minus, q_plus, ...
%     J1, J2, ...
%     x_s, y_s, nx_s, ny_s, hx, hy, sigma, option);
% 
% % get rate
% % J_ab_1_new = rot90(reshape(J_ab_1, nx_s, ny_s));
% % rate = sum(J_ab_1_new(:,ceil(nx_s/2)))*hy;
% % disp(['rate is ',num2str(rate)]);