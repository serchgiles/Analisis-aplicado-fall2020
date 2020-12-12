def grad(f, x, eps=1e-5):
    from numpy import array, zeros
    x = array(x)
    n = x.size
    gr = zeros(n)
    for i in range(n):
        ei = zeros(n)
        ei[i] = 1
        gr[i] = (f(x+ei*eps)-f(x-ei*eps))/(2*eps)
    #endfor
    return gr