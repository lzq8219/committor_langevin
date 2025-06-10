function contourtpt(q, nx_s, ny_s, X, Y, sigma)


Q = rot90(reshape(q, nx_s, ny_s));
%index = find(Q == 0);
% new_Q = Q;
%new_Q(index) = ones(size(index));
para = linspace(0,pi,20);
para = max(q)*(cos(para)+1)/2;

figure, contourf(X, Y, Q, para), colormap(jet), colorbar;
set(gca,'FontSize',20);
xlabel('x1');
ylabel('x2');
title(['sigma = ',num2str(sigma)]);


end