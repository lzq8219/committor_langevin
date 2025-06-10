function b2_value = b2(x, y, a, h, l, scale)
%% Return the second component of -GradV.


	% checked on March 17, 2015.

    % b2 is the second component of b(x) where x\in \mathbb{R}^d.
    % here x and y (input) are scalar, vector or matrix.


%     b2_value = -10*y;


    % triple-well potential.
%     b2_value = 3*exp(-x.^2-(y-1/3).^2).*(-2*(y-1/3)) ...
%         - 3*exp(-x.^2-(y-5/3).^2).*(-2*(y-5/3)) ...
%         - 5*exp(-(x-1).^2-y.^2).*(-2*y) ...
%         - 5*exp(-(x+1).^2-y.^2).*(-2*y) ...
%         + 0.8*(y-1/3).^3;
%     b2_value = -b2_value;
%        

%% Mueller-Brown Surface.

    [~,fY] = CalForce(x,y);
    fY = -fY;
    
    % perturbation term
    cst = 1/(l^2);
    
    if mod(length(a),2) == 0
        M = length(a)/2;
        for k = 1:M
            xi = a(2*k-1);
            yi = a(2*k);
            value = h*exp(-cst/2*( (x-xi)^2+(y-yi)^2 ));
            fY = fY + value*(-cst*(y-yi));
        end
    end
    b2_value = -fY;
    
    
    b2_value = scale*b2_value;
    
end