import numpy as np
import matplotlib.pyplot as plt
datos = np.loadtxt('datos.txt')
Rsq = datos[:,0]
Rsq_new = datos[:,2]
F = datos[:,1]
F_new = datos[:,3]

plt.figure(figsize = (10,10))

plt.subplot(2,2,1)
_= plt.hist(Rsq, bins = 80)
plt.title('Histograma de R^2')
plt.subplot(2,2,2)
_= plt.hist(Rsq_new, bins = 80)
plt.title('Histograma de R^2 nuevo')
plt.subplot(2,2,3)
_= plt.hist(F, bins = 80)
plt.title('Histograma de F')
plt.subplot(2,2,4)
_= plt.hist(F_new, bins = 80)
plt.title('Histograma de F nuevo')

plt.savefig('Grafica.png')