import os
os.environ['THEANO_FLAGS'] = 'device=gpu'

import numpy as np
import theano as th
import theano.tensor as tt

rng = np.random

N = 400         # training sample size
feats = 784     # number of input variables

## generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

x = tt.dmatrix('x')
y = tt.dvector('y')

w = th.shared(rng.randn(feats), name='w')
b = th.shared(0., name='b')

print('Initial model:')
print(w.get_value())
print(b.get_value())

## building computation graph
prob = 1 / (1 + tt.exp(-tt.dot(x, w) - b))
pred = prob > 0.5
# cross-entropy loss function
xent = -y * tt.log(prob) - (1-y) * tt.log(1 - prob)
cost = xent.mean() + 0.01 * (w**2).sum()
gw, gb = tt.grad(cost, [w, b])

## compile graph
train = th.function(
    inputs=[x, y],
    outputs=[pred, xent],
    updates=[(w, w - 0.1*gw), (b, b - 0.1*gb)]
)
predict = th.function(
    inputs=[x],
    outputs=pred
)

## train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print('Final model:')
print(w.get_value())
print(b.get_value())
print('# target values on D:')
print(D[1])
print('# prediction on D:')
p = predict(D[0])
print(p)
print('# differences:')
print(np.abs(p - D[1]))