%% Documentation. 
% The equation is 
% \dot{X}_t = P_t
% \dot{V}_t = -\grad U(X_t) - \gamma p_t + \sqrt{2\gamma k_BT} dW_t.
% a = sigma^2 I.

addpath(genpath(pwd));

kbt = 1;
gamma = 1;

xmin = -4.0;
xmax = 3.0;
ymin = -2.5;
ymax = 4.5;
p1min = -3;
p1max = 3;
p2min = -3;
p2max = 3;

nx = 100;
ny = 100;
np1 = 10;
np2 = 10;

hx = (xmax-xmin)/nx;
hy = (ymax-ymin)/ny;
hp1 = (p1max-p1min)/np1;
hp2 = (p2max-p2min)/np2;
x_whole = linspace(xmin, xmax, nx+1); % the whole space
y_whole = linspace(ymin, ymax, ny+1);
p1_whole = linspace(p1min, p1max, np1+1); 
p2_whole = linspace(p2min, p2max, np2+1);

x_s = x_whole(2:nx); %truncated x (shorted).
y_s = y_whole(2:ny); %truncated y (shorted).
p1_s = p1_whole(2:np1); %truncated x (shorted).
p2_s = p2_whole(2:np2); %truncated y (shorted).
nx_s = nx-1; %shorted nx.
ny_s = ny-1; %shorted ny.
np1_s = np1-1; %shorted nx.
np2_s = np2-1; %shorted ny.

option = 0.4;

CalGrad(0,0)

f = @(x,y,p1,p2)in_a([x,y],option)*1+~in_a([x,y],option)*0;
grad_U_fun = @(x,y)CalGrad(x,y);

% Start the timer  
tic;  

% Code block to measure  
q=back_kol_solve_4d(x_s, y_s, p1_s, p2_s,...
    nx_s, ny_s, np1_s, np2_s, hx, hy, hp1, hp2, gamma, kbt,...
    grad_U_fun, @(x)in_a(x, option), @(x)in_b(x,option), f);  

% Stop the timer and get the elapsed time  
elapsedTime = toc;  

% Display the elapsed time  
disp(['Elapsed time: ', num2str(elapsedTime), ' seconds']);  
