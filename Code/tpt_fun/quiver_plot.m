function quiver_plot(u, v, x_s, y_s, nx_s, ny_s)

% checked on March 18, 2015.

[X, Y] = meshgrid(x_s, y_s);
Y = flipud(Y);

u_new = rot90(reshape(u, nx_s, ny_s));
v_new = rot90(reshape(v, nx_s, ny_s));

select_x = 1:floor(nx_s/40):nx_s; % the quiver plot cannot do it for too dense vector field.
select_y = 1:floor(ny_s/40):ny_s;

% quiver plot for the vector field.
figure, quiver(X(select_y,select_x),Y(select_y,select_x),...
    u_new(select_y,select_x),v_new(select_y,select_x),'k');

end