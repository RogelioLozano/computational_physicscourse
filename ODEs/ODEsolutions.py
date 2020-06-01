import numpy as np

"""
Module with solvers of ordinary differential equations (ODEs)
"""


###AUXILIAR FUNCTIONS

def isiterable(obj):
    """auxiliar function"""
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True 

#GENERAL SOLVERS OF ODES
    
def sol_System_ODE(f,interval,r0,num_steps,method="euler"):
    """Solving systems of initial value problems. vec{r}' = vec{f}(vec{r},t)
    where vec{r} is a vector of variables constituting the system, and the vector function f being the system
    definition. 

    Assumes that f takes an array r and time float t, f(r,t). f must return an array(r1,r2)
    
    methods: "euler", "RK4", "midpoint","leap_frog"
    
    Input: interval(a tuple with initial and final points of evaluation), a vector of initial values vec{r0} 
    and number of steps.
    output: time steps ti and solution array r_sol
    

    Example:
    
    def f(r,t):
    y1,y2 = r[0],r[1]
    return np.array([y2**2-2*y1, y1-y2-t*y2**2],float)

    interval = (0,1)
    r0 = (0,1)
    N=10
    
    sol_System_ODE(f,(t0,tf),r0,N,method="RK4")

    """
    
    assert type(method)==str and method in ["euler","RK4","midpoint","leap_frog"],\
    "Not a valid method, select one from the options."
    
    assert isiterable(r0), "Must pass an iterable of initial values and functions. If you want to\
    solve single ODE use sol_ODE instead."
    
    h = (interval[1]-interval[0])/num_steps
    time=np.arange(interval[0],interval[1],h)
    r = np.array(r0,float)
    # los resultados se almacena en una lista para que se pueda trabajar con arrays libremente en los calculos.
    r_sol=[[],[]]
    
    def select(method):
        m=method.lower()
        def Euler_step(f,y,t):
            return h*f(y,t)
        
        def RK4_step(f,y,t):
            
            s1 = f(y,t)
            y2 = y + h*s1*0.5
            s2 = f(y2,t+h*0.5)
            y3 = y + h*s2*0.5
            s3 = f(y3,t+h*0.5)
            y4 = y + h*s3
            s4 = f(y4,t+h)
            
            return h*(s1+2*s2+2*s3+s4)/6
        
        def midpoint_step(f,y,t):
            w = f(y,t)
            y1 = y + h*w*0.5   
            
            return h*f(y1,t+h*0.5)

        def leap_frog(f,y,t):
            if abs(t - time[0])<1e-25:
                # we need a global var because we need to keep track of its value to increment it later inside the iteration
                global self_start
                self_start = y + 0.5*Euler_step(f,y,t)
            self_start = self_start + h*f(y,t)
            return  h*f(self_start,t+0.5*h)


        if m == "euler":
            return Euler_step
        elif m == "rk4":
            return RK4_step
        elif m == "midpoint":
            return midpoint_step
        elif m == "leap_frog":
            return leap_frog
    
    metamethod = select(method)
    
    for ti in time:
        r_sol[0].append(r[0])
        r_sol[1].append(r[1])
            
        r+=metamethod(f,r,ti)
        
    return np.array(time),np.array(r_sol)


#####UNIT TESTS

def test():
    import matplotlib.pyplot as plt

    def f(r,t):
        y1,y2 = r[0],r[1]
        return np.array([y2**2-2*y1, y1-y2-t*y2**2],float)

    interval = (0,1)
    r0 = (0,1)
    N=10

    #Analytical solutions 
    analytical=[lambda t: t*np.exp(-2*t),lambda t:np.exp(-t)]

    t,sol = sol_System_ODE(f,interval,r0,N,method="euler")
    plt.plot(t,sol[0],t,sol[1])
    plt.plot(t,analytical[0](t),'--',t,analytical[1](t),'--')
    plt.title("Analytical: -- euler numerical: solid line")
    plt.show()

    t,sol = sol_System_ODE(f,interval,r0,N,method='midpoint')
    plt.plot(t,sol[0],t,sol[1])
    plt.plot(t,analytical[0](t),'--',t,analytical[1](t),'--')
    plt.title("Analytical: -- midpoint numerical: solid line")
    plt.show()

    t,sol = sol_System_ODE(f,interval,r0,N,method='RK4')
    plt.plot(t,sol[0],t,sol[1])
    plt.plot(t,analytical[0](t),'--',t,analytical[1](t),'--')
    plt.title("Analytical: -- RK4 numerical: solid line")
    plt.show()

    return None

if __name__ == "__main__":
    print("This program shows plots of numeric solutions of ODEs using various methods and comparing them\
    with the analytical solution of the system.")
    test()


    