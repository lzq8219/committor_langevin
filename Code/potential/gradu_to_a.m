function result = gradu_to_a(x, y, a, h, l, scale)


cst = 1/(l^2);

result = zeros(size(a));
if mod(length(a),2) == 0
    M = length(a)/2;
    for k = 1:M
        xi = a(2*k-1);
        yi = a(2*k);
        value = h*exp(-cst/2*( (x-xi)^2+(y-yi)^2 ));
        result(2*k-1) = value*(-cst*(xi-x));
        result(2*k) = value*(-cst*(yi-y));
    end
end

result = result*scale;


end