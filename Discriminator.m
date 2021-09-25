classdef Discriminator < handle
   properties
      weights
      bias 
   end
   methods
       function obj = Discriminator()
             obj.weights = normrnd(-1,1,[1,4]);
             obj.bias =normrnd(-1,1);
        end
       
       function g = sigmoid(obj,z)
        g = 1.0 ./ (1.0 + exp(-z));
       end
      function f =forward(obj,x, f)
%          disp("dif")
%         disp(obj.weights)
        if (isequal(f,0))
            f =obj.sigmoid((x.*obj.weights) +obj.bias);
        else
            
            f =obj.sigmoid(sum(x.*obj.weights) +obj.bias);
        end
        
      end
      function e = error(obj,i)
        
        p = obj.forward(i,1) ;
        
        e = -log(p);
        
      end
     function [d_w, d_b]=derivatives(obj,i)
        p = obj.forward(i,1);
        d_w = -i .* (1-p);
        d_b = -(1-p);

     end
     function update(obj,i, lr)
        [derw, derb] = obj.derivatives(i);

        obj.weights = obj.weights -(lr* derw);
        obj.bias = obj.bias -(lr.* derb);
        
     end
     function e = noise_error(obj,n)
        
        p = obj.forward(n,0);
        e = -log(1-p);
      end
        
      function [d_wn, d_bn] =noise_derivatives(obj,n)
        p = obj.forward(n,1); 
        d_wn = n .*p; 
        d_bn = p;
      end

      function noise_update(obj,n, lr, self)
        [d_wn, d_wb] = obj.noise_derivatives(n);
        self.weights = self.weights -(lr.*d_wn);
        self.bias = self.bias -(lr.*d_wb);
        
      end
   end
end