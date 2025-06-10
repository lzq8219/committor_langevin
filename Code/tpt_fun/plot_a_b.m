function plot_a_b(x_s,y_s,nx_s,ny_s,option)

% This function plots the region of set A and set B.
% checked on March 18, 2015.


for i = 1:nx_s
    for j = 1:ny_s
        point = [x_s(i),y_s(j)];
        if in_a(point,option) == 1
            plot(x_s(i),y_s(j),'x','Color',[0.6,0.6,0.6]);
        end
        if in_b(point,option)==1
            plot(x_s(i),y_s(j),'d','Color',[.3,.3,.3]);
        end
    end

end

end