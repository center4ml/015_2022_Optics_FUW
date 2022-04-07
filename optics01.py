import math
import matplotlib.pyplot as plt
import scipy.special
import torch

points = 256

time = torch.arange(-10., 10., 20. / points)

source_t = torch.exp(-time ** 2 / 2.)
source_f = torch.fft.fft(source_t)

hermite = scipy.special.hermite(1)
target_t = torch.exp(-time ** 2 / 2.) * torch.tensor(hermite(time))

phase_f = torch.rand(points, requires_grad = True)

optimizer = torch.optim.Adam([phase_f])
for epoch in range(10000):
    result_t = torch.fft.ifft(source_f * torch.exp(2. * math.pi * 1j * phase_f)).real
    loss = torch.square(result_t / result_t.std() - target_t / target_t.std()).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch, loss.item())

result_t = result_t.detach()

plt.plot(time, target_t / target_t.std(), '.')
plt.plot(time, result_t / result_t.std())

plt.show()

