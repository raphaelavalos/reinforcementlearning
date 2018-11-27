import numpy as np
from linearmab_models import *
import algorithm
import matplotlib.pyplot as plt

random_state = np.random.randint(0, 24532523)

model = ColdStartMovieLensModel(
    random_state=random_state,
    noise=0.1
)

print('best arm : %i' % np.dot(model.features, model.real_theta).argmax())

lucb = algorithm.LinearUCB(model, lamb=1, alpha=100, T=6000)
a = lucb.average(30)
regret_lucb = a[:, 1]
diff_theta_lucb = a[:, 2]

random = algorithm.Random(model, T=6000)
b = random.average(30)
regret_rand = b[:, 1]
diff_theta_rand = b[:, 2]

epsilon = algorithm.Epsilon(model, T=6000, epsilon=0.90, lamb=1)
c = epsilon.average(30)
regret_epsilon = c[:, 1]
diff_theta_epsilon = c[:, 2]

epsilon2 = algorithm.Epsilon(model, T=6000, epsilon=0.80, lamb=1)
d = epsilon2.average(30)
regret_epsilon2 = d[:, 1]
diff_theta_epsilon2 = d[:, 2]

plt.plot(np.cumsum(regret_lucb))
plt.plot(np.cumsum(regret_rand))
plt.plot(np.cumsum(regret_epsilon))
plt.plot(np.cumsum(regret_epsilon2))
plt.legend(['LUCB', 'Random', 'Epsilon 0.9', 'Epsilon 0.8'])
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.yscale('log')
plt.show()

plt.plot(diff_theta_lucb)
plt.plot(diff_theta_rand)
plt.plot(diff_theta_epsilon)
plt.plot(diff_theta_epsilon2)
plt.legend(['LUCB', 'Random', 'Epsilon 0.9', 'Epsilon 0.8'])
plt.xlabel('Rounds')
plt.ylabel('||theta* - theta||2')
# plt.yscale('log')
plt.show()
