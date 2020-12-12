def optimal_test(f, x):
    from numpy.linalg import eigvals, norm
    from grad import grad
    if norm(grad(f,x)) < 1e-5 :
        from hess import hess
        eigenval = eigvals(hess(f,x))
        if (eigenval > 0).all():
            return True
        else :
            print("Hessian matrix is not positive definite")
            return False
        #endif
    #endif
    else :
        print("Gradient is not zero")
        return False
        