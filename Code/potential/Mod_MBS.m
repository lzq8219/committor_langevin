function V = Mod_MBS(x, y, a, h, l)

% Modified Mueller-Brown Surface.

    V = CalPote(x, y);
    
    cst = 1/(l^2);
    
    if mod(length(a),2) == 0
        M = length(a)/2;
        for k = 1:M
            xi = a(2*k-1);
            yi = a(2*k);
            value = h*exp(-cst/2*( (x-xi)^2+(y-yi)^2 ));
            
            V = V+value;
        end
    end

end