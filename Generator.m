classdef Generator <handle
   properties
      weights
      bias 
   end
   methods
       function obj = Generator()
             obj.weights = normrnd(-1,1,[1,4]);
             obj.bias =normrnd(-1,1,[1,4]);
        end
       
       function g = sigmoid(obj,z)
        g = 1.0 ./ (1.0 + exp(-z));
       end
      function f =forward(obj,x)
        f =obj.sigmoid(x.*obj.weights+obj.bias);
      end
      function e = error(obj,i,D)
        
        p = obj.forward(i);
        y=D.forward(p,1);
        e = -log(y);
      end
     function [d_w, d_b]=derivatives(obj,i,D)
        diw =D.weights;
        dib =D.bias;
        x = obj.forward(i);
        y = D.forward(x,1);
        factor = -(1-y) .* diw .* x .*(1-x);
        d_w =factor .*i;
        d_b = factor;
     end
     function update(obj,i,D, lr)
        e_b = obj.error(i,D);
        [derivw, derivb] = obj.derivatives(i,D);
        obj.weights = obj.weights -(lr .* derivw);
        obj.bias = obj.bias -(lr .* derivb);
        e_d = obj.error(i,D);
     end
     

   end
end