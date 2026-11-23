# 23-11-2026
## NOTAS NICO:
- Representar no solo la gráfica de pérdida de train, sino representar la de validación en paralelo, para que cuando la métrica de validación empiece a subir cortar el entrenamiento ya que eso significa que está habiendo overfitting
- Hacer un mecanismo de early stopping para que cuando la validación siga subiendo durante 4-5 épocas se corte el entrenamiento
- hacer un entrenamiento completo del modelo en lugar de solo la cabeza
- Quitar los datos de train al completo 
- No tocar para nada los datos de test