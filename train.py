import numpy as np
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegression

# Entrenamiento del modelo
X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
y = np.array([0, 0, 1, 1, 1, 0])
lr = LogisticRegression()
lr.fit(X, y)

# Evaluación del modelo
score = lr.score(X, y)
print(f"Score: {score}")

# Predicciones con el modelo entrenado
predictions = lr.predict(X)
print(f"Predictions: {predictions}")

# Hacer predicciones con nuevos datos
new_data = np.array([3, 4, -3, 0]).reshape(-1, 1)
new_predictions = lr.predict(new_data)
print(f"New Predictions: {new_predictions}")

# Guardar las nuevas predicciones en un archivo
np.savetxt("new_predictions.csv", new_predictions, delimiter=",", fmt="%d")
