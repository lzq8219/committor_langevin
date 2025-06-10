function stop = outfun(x,optimValues,state)
    global sequence;
    global M;
    
    stop = false;
    
 
     switch state
         case 'init'
 
         case 'iter'
         % Concatenate current point and objective function
         % value with history. x must be a row vector.
       
           sequence=[sequence ; [x(1:2*M) optimValues.fval]];
         
         case 'done'
       
         otherwise
     end
 end