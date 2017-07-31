import numpy as np
import matplotlib.pyplot as plt


def Shields_critical(D, g=9.806, ro_s=2650, ro_w=1027, v=1.36*10**(-6)):
    s = ro_s / ro_w
    D_star = (g * (s - 1) / v ** 2) ** (1 / 3) * D
    return 0.3 / (1 + 1.2 * D_star) + 0.055 * (1 - np.exp(-0.02 * D_star))


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
with plt.style.context('bmh'):
    plt.plot(np.arange(0.1, 10, 0.01), Shields_critical(D=np.arange(0.1, 10, 0.01) / 1000), lw=2, color='orangered')
    plt.semilogx()
    plt.title(r'\textbf{Critical Shields parameter for $\rho_s=2650$}')
    plt.xlabel(r'\textbf{Particle diameter} \textit{(mm)}')
    plt.ylabel(r'\textbf{Critical Shields parameter}')
    plt.savefig(r'D:\Work folders\desktop projects\1 Living breakwaters\shields.png', dpi=300, bbox_inches='tight')
    plt.close()
