function V = potential_V(x, y, a, h, l, scale)


	% checked on March 17, 2015.


%     V = 5/2*(x.^2-1).^2 + 5*y.^2;
    
%     V = Triple_Pote(x, y);




    V = scale*Mod_MBS(x, y, a, h, l);

end