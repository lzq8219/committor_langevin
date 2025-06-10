function b1_value = b1(x, y, a, h, l, scale)

%% Return the first component of -Grad V.
	
    % checked on March 17, 2015.



    % b1 is the first component of b(x) where x\in \mathbb{R}^d.
    % here x and y (input) are scalar, vector or matrix.


%     b1_value = -10*x.*(x.^2-1);


    % triple-well potential.
%     b1_value = 3*exp(-x.^2-(y-1/3).^2).*(-2*x) ...
%         - 3*exp(-x.^2-(y-5/3).^2).*(-2*x) ...
%         - 5*exp(-(x-1).^2-y.^2).*(-2*(x-1)) ...
%         - 5*exp(-(x+1).^2-y.^2).*(-2*(x+1)) ...
%         + 0.8*x.^3;
%     b1_value = -1*b1_value; 

%% Mueller-Brown Surface
    
    [fX, ~] = CalForce(x,y);
    fX = -fX;
    
    % perturbation term
    cst = 1/(l^2);
    
    if mod(length(a),2) == 0
        M = length(a)/2;
        for k = 1:M
            xi = a(2*k-1);
            yi = a(2*k);
            value = h*exp(-cst/2*( (x-xi)^2+(y-yi)^2 ));
            fX = fX + value*(-cst*(x-xi));
        end
    end
    b1_value = -fX;
    

    b1_value = scale*b1_value;
end