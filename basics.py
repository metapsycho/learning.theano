import os
os.environ['THEANO_FLAGS'] = 'device=cpu'

import numpy as np
import theano as th
import theano.tensor as tt
from theano import function
from theano import pp
from theano import shared
from theano.ifelse import ifelse
import time

th.config.floatX = 'float64'
th.config.mode = 'FAST_COMPILE'

# Baby Steps - Algebra
if False:
    print('--------------------')
    x = tt.dscalar('x') # declare a double scalar
    y = tt.dscalar('y')
    z = x + y
    f = function([x, y], z)
    print(f(2, 3))

if False:
    print('--------------------')
    x = tt.dmatrix('x')
    y = tt.dmatrix('y')
    z = x + y
    f = function([x, y], z)
    print(f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]])))

# More Examples
if False:
    print('--------------------')
    a, b = tt.dmatrices('a', 'b')
    diff = a - b
    diff_abs = abs(diff)
    diff_squared = diff**2
    f = function([a, b], [diff, diff_abs, diff_squared])
    print(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))

if False:
    print('--------------------')
    state0 = shared(0)
    state1 = shared(0)
    inc = tt.iscalar('inc')
    #accumulator = function([inc], state0, updates={state0: state0+inc, state1: state1-inc})
    accumulator = function([inc], [state0, state1], updates=[(state0, state0+inc), (state1, state1-inc)])
    print(accumulator(1))
    state1.set_value(0)
    print(accumulator(300))
    print(state0.get_value(), state1.get_value())
    
if False:
    print('--------------------')
    state = shared(0)
    inc = tt.iscalar('inc')
    fn_of_state = state*2 + inc
    foo = tt.scalar(dtype=state.dtype)
    skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)], updates=[(state, state+1)])
    print('state: ', state.get_value())
    print(skip_shared(1, 3))
    print('state: ', state.get_value())
    print(skip_shared(1, 3))
    print('state: ', state.get_value())

# Derivatives in Theano
if False:
    print('--------------------')
    x = tt.dmatrix('x')
    y = tt.sum(1 / (1 + tt.exp(-x)))
    gy = tt.grad(y, x)
    f = th.function([x], gy)
    print(pp(gy))
    print(pp(f.maker.fgraph.outputs[0]))

if False:
    print('--------------------')
    x = tt.dvector('x')
    y = x**2
    j = th.gradient.jacobian(y, x)
    f = function([x], j)
    #print(pp(j))
    print(f([4, 4]))
    h = th.gradient.hessian(y.sum(), x)
    g = function([x], h)
    #print(pp(h))
    print(g([4, 4]))
    
if False:
    print('--------------------')
    x = tt.dvector('x')
    y = x**2
    J, updates = th.scan(
        lambda i,y,x: tt.grad(y[i], x),
        sequences=tt.arange(y.shape[0]),
        non_sequences=[y, x])
    f = function([x], J, updates=updates)
    print(f([4, 4]))

if False:
    print('--------------------')
    x = tt.dvector('x')
    y = (x**2).sum()
    gy = tt.grad(y, x)
    H, updates = th.scan(
        lambda i,gy,x: th.grad(gy[i], x),
        sequences=tt.arange(gy.shape[0]),
        non_sequences=[gy, x])
    f = function([x], H, updates=updates)
    print(f([4, 4]))
    
# Conditions
if False:
    print('--------------------')
    a, b = tt.scalars('a', 'b')
    x, y = tt.matrices('x', 'y')
    
    f_switch = function(
        [a, b, x, y],
        tt.switch(tt.lt(a, b), tt.mean(x), tt.mean(y)),
        mode=th.Mode(linker='vm'))
    f_ifelse = function(
        [a, b, x, y],
        ifelse(tt.lt(a, b), tt.mean(x), tt.mean(y)),
        mode=th.Mode(linker='vm'))
    
    val1 = 0.
    val2 = 1.
    big_mat1 = np.ones((10000, 1000))
    big_mat2 = np.ones((10000, 1000))
    
    n_times = 10
    
    tic = time.clock()
    for i in range(n_times):
        f_switch(val1, val2, big_mat1, big_mat2)
    print('time spent evaluating both values %f sec' % (time.clock() - tic))
    
    tic = time.clock()
    for i in range(n_times):
        f_ifelse(val1, val2, big_mat1, big_mat2)
    print('time spent evaluating both values %f sec' % (time.clock() - tic))
    
# Loop
if False:
    print('--------------------')
    X = tt.matrix('X')
    W = tt.matrix('W')
    B = tt.vector('B')
    
    results, updates = th.scan(lambda v: tt.tanh(tt.dot(v,W) + B), sequences=X)
    f = function(inputs=[X,W,B], outputs=results)
    
    x = np.eye(2, dtype=th.config.floatX)
    w = np.ones((2, 2), dtype=th.config.floatX)
    b = np.ones((2), dtype=th.config.floatX)
    b[1] = 2
    print('tanh(x(t)*W + b)\n', f(x, w, b))
    
if True:
    print('--------------------')
    coeffs = tt.vector('coeffs')
    X = tt.scalar('X')
    
    poly, _ = th.scan(
        lambda i, f, x: coeffs[i] + f*x,
        sequences=tt.arange(coeffs.size),
        outputs_info=tt.zeros_like(X),
        non_sequences=[X],
        go_backwards=True)
    f = function([coeffs, X], poly)
    test_coeffs = np.asarray([1, 0, 2], dtype=th.config.floatX)
    print(f(test_coeffs, 3))