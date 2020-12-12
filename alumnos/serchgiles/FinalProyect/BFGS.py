def BFGS(f,x0,tol = 1e-4, maxiter = 5000):
    from grad import grad
    from hess import hess
    from line_search import line_search
    from numpy import dot, array, identity, transpose
    from numpy.linalg import solve, norm, eigvals
    from choleskyHes import cholesky_hessian
    
    xk = array(x0, dtype = 'float64')
    n = xk.size
    iterations = 0
    I = identity(n)
    Bk = hess(f,xk)
    if (min(eigvals(Bk)) <= 0):
        Bk = cholesky_hessian(Bk)
    
    while norm(grad(f,xk))>tol and iterations < maxiter:
        iterations += 1
        
        gf_k = grad(f,xk)
        pk = solve(Bk, -gf_k)
        
        alpha = line_search(f,xk,pk)
        
        sk = alpha*pk
        
        yk = grad(f,xk + sk) - grad(f,xk)
        
        
        yk_sk = array([sk]) @ transpose(array([yk]))
        yk_yk = transpose(array([yk])) @ array([yk])
        sBs = (array([sk]) @ Bk) @ transpose(array([sk]))
        Bs = (Bk @ transpose(array([sk])))
        sB = array([sk]) @ Bk
            
        Bk = Bk + yk_yk/yk_sk - (Bs @ sB)/sBs
        
        xk = xk + sk 
    #end while
    return [xk, f(xk), iterations]