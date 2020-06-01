
def errorOf(x):
    """Compute the error of x as defined in Newman as a 
    uniformly distributed random number with standard deviation = C*x
    with C = 10^(-16)
     """
    C = 1e-16
    return C*x

def errorOfaddition(L):
    """Assumes L is a np.array with the numbers to sum,
    it returns the standard deviation (error) of their sum """
    from numpy import sqrt
    errores=errorOf(L)
    return sum(errores**2)

def errorOfproduct(L):
    """Assumes L an np.array of numbers to be multiplied"""
    C = 1e-16
    from numpy import sqrt,product
    x=product(L)
    return sqrt(len(L))*C*x

def errorOfdivision(x1,x2):
    """Assumes x1 and x2 floats and returns error of their division"""
    C = 1e-16
    from numpy import sqrt
    x = x1/x2
    return sqrt(2)*C*x

if __name__=="main":
    print(errorOf(123456543))