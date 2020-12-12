def hess_grad(f,x,eps=1e-6):
    from grad import grad
    from numpy import array, zeros
    x = array(x)
    n = x.size
    H = zeros((n,n))
    for i in range(n):
        ei = zeros(n)
        ei[i] = 1
        H[:,i] = (grad(f, x + eps*ei, eps=eps/2) - grad(f, x - eps*ei, eps = eps/2))/(2*eps)
    #endfor
    return H
            
def hess(f,x,eps=1e-6):
    from grad import grad
    from numpy import array, zeros

    x = array(x)
    n = x.size
    H = zeros((n,n))
    fx = f(x)
    for i in range(n):
        ei = zeros(n)
        ei[i] = 1
        fxi = f(x + eps*ei)
        fxi_m = f(x - eps*ei)
        H[i,i] = (fxi - 2*fx + fxi_m)/(eps**2)
        for j in range(i,n):
            ej = zeros(n)
            ej[j] = 1
            fxij = f(x + eps*(ei+ej))
            fxj = f(x + eps*ej)
            H[i,j] = (fxij - fxi - fxj + fx)/(eps**2)
            H[j,i] = H[i,j]
        #endfor j
    #endfor i
    return H