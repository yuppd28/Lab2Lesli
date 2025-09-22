import numpy as np
import matplotlib.pyplot as plt


def simulate_leslie(L, N0, years=50):
    n_classes = len(N0)
    N = np.zeros((years + 1, n_classes))
    N[0] = N0

    for t in range(years):
        N[t + 1] = L @ N[t]

    return N


# Варіант 1
L1 = np.array([
    [0.3, 2.5, 3.7, 0.3],
    [0.5, 0, 0, 0],
    [0, 0.9, 0, 0],
    [0, 0, 0.75, 0]
])
N0_1 = np.array([100, 65, 78, 140])

# Варіант 2
L2 = np.array([
    [0.4, 2.6, 3.8, 0.4],
    [0.4, 0, 0, 0],
    [0, 0.8, 0, 0],
    [0, 0, 0.65, 0]
])
N0_2 = np.array([900, 630, 745, 910])

# Симуляція
N1 = simulate_leslie(L1, N0_1)
N2 = simulate_leslie(L2, N0_2)


# Візуалізація
def plot_population(N, title):
    years = np.arange(N.shape[0])
    plt.figure(figsize=(10, 6))
    for i in range(N.shape[1]):
        plt.plot(years, N[:, i], label=f"Вікова група {i}")
    plt.plot(years, N.sum(axis=1), 'k--', label="Загальна чисельність")
    plt.title(title)
    plt.xlabel("Роки")
    plt.ylabel("Кількість особин")
    plt.legend()
    plt.grid()
    plt.show()


plot_population(N1, "Динаміка популяції (Варіант 1)")
plot_population(N2, "Динаміка популяції (Варіант 2)")

# Аналіз стійкості
eigvals1 = np.linalg.eigvals(L1)
eigvals2 = np.linalg.eigvals(L2)
print("λ (Варіант 1):", max(np.abs(eigvals1)))
print("λ (Варіант 2):", max(np.abs(eigvals2)))
