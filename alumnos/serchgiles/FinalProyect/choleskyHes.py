def cholesky_hessian(A, maxiter=10000):
    from numpy import diag, identity, dot
    from numpy.linalg import cholesky
    
    beta=10**-3
    amin = min(diag(A))
    n = A.shape[0]
    I = identity(n)
    if amin > 0 :
        t = 0
    else:
        t = (-1)*amin + beta
    for i in range(maxiter):
        try:
            L = cholesky(A + t*I)
            break
        except:
            t = max(2*t,beta)
    return dot(L,L)