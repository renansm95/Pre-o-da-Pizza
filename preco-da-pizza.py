from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

x = np.array([6, 8, 10, 14, 18]) ## diâmetro da pizza em polegadas
y = np.array([7, 9, 13, 16, 20]) ## preço da pizza em reais

plt.scatter(x, y)
plt.xlabel('Diâmetro em Polegadas')
plt.ylabel('Preço da pizza em R$')
plt.title('Preço da pizza em razão do seu diâmetro')
plt.show()

model = LinearRegression()
x = x.reshape(-1, 1)
model.fit(x, y)

y_pred = model.predict(x)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R2: {r2:.2f}')

x_new = np.array([12])
y_new = model.predict(x_new.reshape(-1, 1))
print(f'Uma pizza de {x_new} polegadas custaria R$: {y_new[0]:.2f}')