function [q1,q2] = gradient_q( q, nx_s, ny_s, hx, hy)


% checked on March 18, 2015.


% This function gives the gradient of committor function q.
% q could be q^{+} or q^{-}.


% (q1,q2)=\nabla q.
q1 = zeros(nx_s*ny_s,1);
q2 = zeros(nx_s*ny_s,1);

for i = 1:nx_s
    for j = 1:ny_s

        idx = (j-1)*nx_s+i;
        if i == 1
            q1(idx) = (q(idx+1)-q(idx))/(2*hx);
        elseif i == nx_s
            q1(idx) = (q(idx)-q(idx-1))/(2*hx);
        else
            q1(idx) = (q(idx+1)-q(idx-1))/(2*hx);
        end
        
        if j == 1
            q2(idx) = (q(idx+nx_s)-q(idx))/(2*hy);
        elseif j == ny_s 
            q2(idx) = (q(idx)-q(idx-nx_s))/(2*hy);
        else
            q2(idx) = (q(idx+nx_s)-q(idx-nx_s))/(2*hy);
        end

    end
end

end
