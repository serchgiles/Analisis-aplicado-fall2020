def newton_fixed_step_method(f, x0, tol = 1e-5, maxiter = 10000):
    from grad import grad
    from hess import hess
    from numpy import dot, array, identity
    from numpy.linalg import solve, norm
    
    xk = array(x0, dtype = 'float64')
    n = xk.size
    iterations = 0
    alpha = 0.5
    I = identity(n)
    while norm(grad(f,xk))>tol and iterations < maxiter:
        iterations += 1

        Hinv = solve(hess(f,xk), I)
        gf_k = grad(f,xk)
        pk = (-1)*dot(Hinv,gf_k)
        
        xk += alpha*pk
        
    #end while
    return [xk, f(xk), iterations]

def newton_BLS_method(f, x0, tol = 1e-5, maxiter = 10000):
    from grad import grad
    from hess import hess
    from line_search import line_search
    from numpy import dot, array, identity
    from numpy.linalg import solve, norm
    
    xk = array(x0, dtype = 'float64')
    n = xk.size
    iterations = 0
    I = identity(n)
    while norm(grad(f,xk))>tol and iterations < maxiter:
        iterations += 1

        Hinv = solve(hess(f,xk), I)
        gf_k = grad(f,xk)
        pk = (-1)*dot(Hinv,gf_k)
        alpha = line_search(f,xk,pk)
        xk += alpha*pk
        
    #end while
    return [xk, f(xk), iterations]

def newton_modHes_method(f, x0, tol = 1e-5, maxiter = 10000):
    from grad import grad
    from hess import hess
    from line_search import line_search
    from numpy import dot, array, identity
    from numpy.linalg import solve, norm, eigvals
    from choleskyHes import cholesky_hessian
    
    xk = array(x0, dtype = 'float64')
    n = xk.size
    iterations = 0
    I = identity(n)
    while norm(grad(f,xk))>tol and iterations < maxiter:
        iterations += 1

        H = hess(f,xk)
        if (min(eigvals(H)) <= 0):
            H = cholesky_hessian(H)
        gf_k = grad(f,xk)
        pk = solve(H, -gf_k)
        alpha = line_search(f,xk,pk)
        xk += alpha*pk
        
    #end while
    return [xk, f(xk), iterations]