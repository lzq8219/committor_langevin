function [J_ab_1, J_ab_2] = get_j_ab(m, q_minus, q_plus, J1, J2, ...
    x_s, y_s, nx_s, ny_s, hx, hy, sigma, option)

% checked on March 18, 2015.

% This function is to obtain J^{AB}, the probability flux.

[q_minus_1, q_minus_2] = gradient_q(q_minus, nx_s, ny_s, hx, hy);
[q_plus_1, q_plus_2] = gradient_q(q_plus, nx_s, ny_s, hx, hy);


J_ab_1 = q_minus.*q_plus.*J1+...
    sigma^2*m.*(q_minus.*q_plus_1-q_plus.*q_minus_1);
J_ab_2 = q_minus.*q_plus.*J2+...
    sigma^2*m.*(q_minus.*q_plus_2-q_plus.*q_minus_2);

quiver_plot(J_ab_1, J_ab_2, x_s, y_s, nx_s, ny_s);
hold on; plot_a_b(x_s,y_s,nx_s,ny_s,option);

end