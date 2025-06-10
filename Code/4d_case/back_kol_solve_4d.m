function q = back_kol_solve_4d(x_s, y_s, p1_s, p2_s,...
    nx_s, ny_s, np1_s, np2_s, hx, hy, hp1, hp2, gamma, kbt,...
    grad_U_fun, in_a_fun, in_b_fun, f)

    idx_i = zeros(1, 9*nx_s*ny_s*np1_s*np2_s); % store information related to row.
    idx_j = zeros(1, 9*nx_s*ny_s*np1_s*np2_s); % col in sparse matrix.
    idx_k = zeros(1, 9*nx_s*ny_s*np1_s*np2_s); % value in sprase matrix.
    idx_count = 0; % count the number of parameter sets for sparse matrix.
    in_a_count = 0; % count the number of points in Set A.
    in_b_count = 0; % count the number of points in Set B.
    b_vec = zeros(nx_s*ny_s*np1_s*np2_s,1);
    
    
    for i = 1:nx_s
        for j = 1:ny_s
            for k = 1 : np1_s
                for l = 1:np2_s
                    
                    idx = (l-1) * np1_s * nx_s * ny_s + (k-1) * nx_s * ny_s+(j-1)*nx_s+i;
                    b_vec(idx) = f(x_s(i),y_s(j),p1_s(k),p2_s(l));
                    idx_in_sparse_matrix = 9*(idx-1);
                    itself_repo = 0;
                    
                    if in_a_fun([x_s(i),y_s(j)]) == 1
                        idx_count = idx_count+1;
                        idx_i(idx_in_sparse_matrix+1) = idx;
                        idx_j(idx_in_sparse_matrix+1) = idx;
                        idx_k(idx_in_sparse_matrix+1) = 1;
                        in_a_count = in_a_count+1;
                    elseif in_b_fun([x_s(i),y_s(j)]) == 1
                        idx_count = idx_count+1;
                        idx_i(idx_in_sparse_matrix+1) = idx;
                        idx_j(idx_in_sparse_matrix+1) = idx;
                        idx_k(idx_in_sparse_matrix+1) = 1;
                        in_b_count = in_b_count+1;
                    else
                        bb = grad_U_fun(x_s(i),y_s(j));
                        b1 = bb(1);
                        b2 = bb(2);
                        %update right
                        if i~=nx_s
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+2) = idx;
                            idx_j(idx_in_sparse_matrix+2) = idx+1;
                            idx_k(idx_in_sparse_matrix+2) = p1_s(k)/2/hx;
                        else
                            itself_repo = itself_repo+...
                                p1_s(k)/2/hx;
                        end
                        %update left
                        if i~=1
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+3) = idx;
                            idx_j(idx_in_sparse_matrix+3) = idx-1;
                            idx_k(idx_in_sparse_matrix+3) = -p1_s(k)/2/hx;
                        else
                           itself_repo = itself_repo+...
                               -p1_s(k)/2/hx;
                        end
                        %update up
                        if j~=ny_s
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+4) = idx;
                            idx_j(idx_in_sparse_matrix+4) = idx+nx_s;
                            idx_k(idx_in_sparse_matrix+4) = p2_s(l)/2/hy;
                        else
                            itself_repo = itself_repo+...
                                p2_s(l)/2/hy;
                        end
                        %update down
                        if j~=1
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+5) = idx;
                            idx_j(idx_in_sparse_matrix+5) = idx-nx_s;
                            idx_k(idx_in_sparse_matrix+5) = -p2_s(l)/2/hy;
                        else
                            itself_repo = itself_repo+...
                                -p2_s(l)/2/hy;
                        end
                        if k~=np1_s
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+6) = idx;
                            idx_j(idx_in_sparse_matrix+6) = idx+nx_s*ny_s;
                            idx_k(idx_in_sparse_matrix+6) = (b1+...
                                gamma*p1_s(k))/(2*hp1) + gamma*kbt/hp1^2;
                        else
                            itself_repo = itself_repo+...
                                (b1+...
                                gamma*p1_s(k))/(2*hp1) + gamma*kbt/hp1^2;
                        end
                        %update left
                        if k~=1
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+7) = idx;
                            idx_j(idx_in_sparse_matrix+7) = idx-nx_s*ny_s;
                            idx_k(idx_in_sparse_matrix+7) = -(b1+...
                                gamma*p1_s(k))/(2*hp1) + gamma*kbt/hp1^2;
                        else
                           itself_repo = itself_repo+...
                               -(b1+...
                                gamma*p1_s(k))/(2*hp1) + gamma*kbt/hp1^2;
                        end
    
                        if l~=np2_s
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+8) = idx;
                            idx_j(idx_in_sparse_matrix+8) = idx+nx_s*ny_s*np1_s;
                            idx_k(idx_in_sparse_matrix+8) = (b2+...
                                gamma*p2_s(l))/(2*hp2) + gamma*kbt/hp2^2;
                        else
                            itself_repo = itself_repo+...
                                (b2+...
                                gamma*p2_s(l))/(2*hp2) + gamma*kbt/hp2^2;
                        end
                        %update left
                        if l~=1
                            idx_count = idx_count+1;
                            idx_i(idx_in_sparse_matrix+9) = idx;
                            idx_j(idx_in_sparse_matrix+9) = idx-nx_s*ny_s;
                            idx_k(idx_in_sparse_matrix+9) = -(b2+...
                                gamma*p2_s(l))/(2*hp2) + gamma*kbt/hp2^2;
                        else
                           itself_repo = itself_repo+...
                               -(b2+...
                                gamma*p2_s(l))/(2*hp2) + gamma*kbt/hp2^2;
                        end
                        
                        %update itself
                        idx_count = idx_count+1;
                        idx_i(idx_in_sparse_matrix+1) = idx;
                        idx_j(idx_in_sparse_matrix+1) = idx;
                        idx_k(idx_in_sparse_matrix+1) = ...
                            kbt*gamma*(-2/hp1^2-2/hp2^2)+itself_repo;
                    end
                end
            
            end       
        end
    end

    
    A_mat = sparse(idx_i(idx_i~=0), idx_j(idx_i~=0), ...
        idx_k(idx_i~=0));
    size(A_mat)
    size(b_vec)
    q = A_mat\b_vec;


end