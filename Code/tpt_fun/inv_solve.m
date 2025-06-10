function m = inv_solve(x_s, y_s, nx_s, ny_s, hx, hy, sigma, b1_fun, b2_fun)


% checked on March 17, 2015.


% This function solves the probability density in tpt.


idx_i = zeros(1,5*nx_s*ny_s);
idx_j = zeros(1,5*nx_s*ny_s);
idx_k = zeros(1,5*nx_s*ny_s);
idx_count = 0;

for i = 1:nx_s
    for j = 1:ny_s
        % update for point x_s(i), y_s(j).
        % since other points at the boundary have value 0.
        % the index for the vector which stores information
        % about m(x) is (j-1)*nx_s+i.
        
        idx = (j-1)*nx_s+i;
        %update right point
        if i~=nx_s
            idx_count = idx_count+1;
            idx_i(idx_count) = idx;
            idx_j(idx_count) = idx+1;
            idx_k(idx_count) = -b1_fun(x_s(i+1),y_s(j))/(2*hx)+sigma^2/(hx^2);
        end
        %update left point
        if i~=1
            idx_count = idx_count+1;
            idx_i(idx_count) = idx;
            idx_j(idx_count) = idx-1;
            idx_k(idx_count) = b1_fun(x_s(i-1),y_s(j))/(2*hx)+sigma^2/(hx^2);
        end
        %update up
        if j~=ny_s
            idx_count = idx_count+1;
            idx_i(idx_count) = idx;
            idx_j(idx_count) = idx+nx_s;
            idx_k(idx_count) = -b2_fun(x_s(i),y_s(j+1))/(2*hy)+sigma^2/(hy^2);
        end
        %update down
        if j~=1
            idx_count = idx_count+1;
            idx_i(idx_count) = idx;
            idx_j(idx_count) = idx-nx_s;
            idx_k(idx_count) = b2_fun(x_s(i),y_s(j-1))/(2*hy)+sigma^2/(hy^2);
        end
        %update itself
        idx_count = idx_count+1;
        idx_i(idx_count) = idx;
        idx_j(idx_count) = idx;
        idx_k(idx_count) = sigma^2*(-2/(hx^2)-2/(hy^2));
    end
end

A_mat = sparse(idx_i(1:idx_count), idx_j(1:idx_count), ...
    idx_k(1:idx_count));
[m, D] = eigs(A_mat, 1, 0);
m = m/(sum(m)*hx*hy); %normalize m.
disp(D);

% plot
% [X, Y] = meshgrid(x_s, y_s);
% Y = flipud(Y);
% M = rot90(reshape(m, nx_s, ny_s)); %m(x) which is consistent with X, Y.
% figure, contour(X, Y, M, 20); %, colormap(gray), colorbar;
% colorbar;

end