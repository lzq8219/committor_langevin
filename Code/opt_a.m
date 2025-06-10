% cd('/home/caoyu/Proj/sensitivity/Code/');
% This is the main script to optimize kappa(a).

% parameters for the tpt.
sigma = 0.5;
xmin = -4.0;
xmax = 3.0;
ymin = -2.5;
ymax = 4.5;
nx = 400;
ny = 400;

% parameters for perturbation part

% a0 = [-0.822, 0.624, 0.212, 0.293];

% paramters for Muller-Brown Surface.
h = 0;
l = 0.25;
scale = 5e-3;


% searching best a.
global M;
M = 1;

a1min = -0.1;  a1max = 0.4;
a2min = 0.1;   a2max = 0.5;

a1min= 0.25; a1max=0.3;
a2min= 0.18; a2max=0.26;

a3min = -1.0;  a3max =  0.0;
a4min =  0.0;  a4max = 0.8;

if (M==1)
	na1 = 10;   na2 = 10;
	a1 = linspace(a1min, a1max, na1);
	a2 = linspace(a2min, a2max, na2);

%% brutal force searching.

alist = zeros(na1*na2, 2*M+1);
for i = 1:na1
  for j = 1:na2
      idx = (j-1)*na1+i;
      alist(idx, 1:(2*M)) = [a1(i), a2(j)];
     
      disp([i,j]);
      [kappa, ~] = getgradkappa([a1(i), a2(j)], sigma, ...
          xmin, xmax, ymin, ymax, nx, ny,...
          h, l, scale, 1)
       alist(idx, 2*M+1) = kappa;  
       
  
       disp([a1(i),a2(j),kappa]);

       pause

  end
end

fig = figure();
 contour(reshape(alist(:,1), na1, na2), ...
     reshape(alist(:,2), na1, na2), ...
     reshape(alist(:,3), na1, na2), 20);
dlmwrite(['result/alist/',num2str(1), '.txt'], alist);
[~,I] = min(alist(:,3));
disp(alist(I,:));
end 

return 

%% use fmincon.
global sequence 
sequence=[];

 options = optimset('GradObj', 'on' , ...
 	'DerivativeCheck', 'off', ...
 	'FunValCheck','on', ...
 	'TolX', 1e-6, 'TolFun', 1e-18, ...
 	'MaxIter', 10, ...
 	'Display','iter', ...
 	 'OutputFcn',@outfun);
 rate = @(a_var) getgradkappa(a_var, sigma, ...
     xmin, xmax, ymin, ymax, nx, ny,...
     h, l, scale, 2);

 if M ==2 
	a_l = [a1min, a2min, a3min, a4min];
 	a_u = [a1max, a2max, a3max, a4max];
	a0 = 0.5*[a1min+a1max, a2min+a2max ... 
			a3min+a3max, a4min+a4max  ];
end 
if M ==1 
	a_l = [a1min, a2min];  	a_u = [a1max, a2max];
	a0 = 0.5*[a1min+a1max, a2min+a2max];	
end 
 [a_star, fval, ~] = fmincon(rate, ...
 	a0,[],[],[],[],a_l,a_u,[],options); 

disp(sequence)



% disp(a_star);
% disp(fval);

% exit;