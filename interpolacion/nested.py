import numpy as np

def nested_mult(coeff,b=[]):
    """Returns polynomial using nested form with Horner’s Method
        Input: array of d+1 coefficients coeff (constant term first),
        array of d base points b, if needed
        Output: vectorized polynomial to be evaluated at x an array or a simple number"""

    def nest(coeff,x,b=b) -> float:
        d = len(coeff)-1
        if b == []:
            b = [0]*d

        y= coeff[-1]
        for i in range(d-1,-1,-1):
            y = y*(x-b[i]) + coeff[i]

        return y

    def poly(coeff):
        def l(b):
            def g(x):
                return nest(x=x,coeff=coeff,b=b)
            return g
        return l
    
    return np.vectorize( poly(coeff)(b) )