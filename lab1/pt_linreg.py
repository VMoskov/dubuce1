import torch
import torch.nn as nn
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # afin regresijski model
    Y_ = a * X + b

    diff = Y - Y_

    # kvadratni gubitak
    loss = torch.mean(diff**2)

    # računanje gradijenata
    loss.backward()

    grad_a_analytic = -2 * torch.mean(diff * X)
    grad_b_analytic = -2 * torch.mean(diff)

    grad_a_toch = a.grad.item()
    grad_b_toch = b.grad.item()

    print(f'step: {i}, loss:{loss}, grad_a_analytic:{grad_a_analytic}, grad_b_analytic:{grad_b_analytic}, grad_a_toch:{grad_a_toch}, grad_b_toch:{grad_b_toch}')

    # korak optimizacije
    optimizer.step()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()