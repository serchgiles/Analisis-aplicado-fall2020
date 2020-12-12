def line_search(f, x, p, c= 1e-4):
    from grad import grad
    from numpy import dot
    from random import uniform
    
    alpha = 1
    while f(x+ alpha*p) > f(x) + c*alpha*dot(grad(f,x),p):
        rho = uniform(0.1,0.9)
        alpha *= rho        
    #end while
    return alpha