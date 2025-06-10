function value = in_b(point, height)


% checked on March 17, 2015.


% if point is in set B, then return 1;
% otherwise, return 0.

%     x = point(1);
%     y = point(2);
    
%     if x > 0 && potential_V(x,y) < height
%         value = 1;
%         return;
%     end
% 
%     value = 0;
%     return;

    % d1 = (x+1)^2+y^2;
    % d2 = (x-1)^2+y^2;
    % d3 = x^2+(y-1.5)^2;
    % d4 = x^2+(y-0.5)^2;

    % if d2 < min([d1,d3,d4]) && potential_V(x,y) < height
    %     value = 1;
    %     return;
    % end

    % value = 0;
    % return;
    
    center = [0.6235, 0.0281];
    r = 0.2;
%     d = (x-1)^2+y^2;
    if norm(point-center,2) <= r
        value = 1;
        return;
    else
        value = 0;
        return;
    end
    
end