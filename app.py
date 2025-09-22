import numpy as np
import matplotlib.pyplot as plt


def simulate_leslie(L, N0, years=50):
    n_classes = len(N0)
    N = np.zeros((years + 1, n_classes))
    N[0] = N0

    for t in range(years):
        N[t + 1] = L @ N[t]

    return N


# --- Варіант 2 ---
L2 = np.array([
    [0.4, 2.6, 3.8, 0.4],  # народжуваність
    [0.4, 0, 0, 0],  # виживання 0->1
    [0, 0.8, 0, 0],  # виживання 1->2
    [0, 0, 0.65, 0]  # виживання 2->3
])

N0_2 = np.array([900, 630, 745, 910])  # початкові кількості

# Симуляція на 50 років
N2 = simulate_leslie(L2, N0_2)

# Побудова графіку
years = np.arange(N2.shape[0])
plt.figure(figsize=(10, 6))
for i in range(N2.shape[1]):
    plt.plot(years, N2[:, i], label=f"Вікова група {i}")
plt.plot(years, N2.sum(axis=1), 'k--', label="Загальна чисельність")
plt.title("Динаміка популяції мишей (Варіант 2)")
plt.xlabel("Роки")
plt.ylabel("Кількість особин")
plt.legend()
plt.grid()
plt.show()

# Аналіз стійкості (головне власне значення λ)
eigvals2 = np.linalg.eigvals(L2)
lambda2 = max(np.abs(eigvals2))
print("Домінантне власне значення λ (Варіант 2):", lambda2)
if lambda2 > 1:
    print("➡ Популяція має тенденцію до зростання.")
elif lambda2 < 1:
    print("➡ Популяція має тенденцію до вимирання.")
else:
    print("➡ Популяція стабільна.")
