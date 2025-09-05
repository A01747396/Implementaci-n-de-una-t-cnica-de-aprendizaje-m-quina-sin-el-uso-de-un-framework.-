# Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
te proyecto implementa una red neuronal feedforward para clasificación binaria, programada desde cero en Python utilizando únicamente NumPy y matplotlib. El objetivo es comprender el funcionamiento interno de una red neuronal, sin depender de frameworks de alto nivel como TensorFlow o PyTorch.

🚀 Características principales

Generación de un dataset sintético balanceado con scikit-learn.
Implementación manual de:

   📌 Forward Propagation
    
   📌 Función de pérdida (Cross-Entropy)
    
   📌 Backpropagation (regla de la cadena, derivadas parciales)
    
   Entrenamiento con ajuste de pesos y sesgos mediante Gradient Descent.
    
   Evaluación con accuracy, loss, classification report y matriz de confusión.
    
   Gráficas de curvas de aprendizaje (train vs validation).
    
   Función de predicción para nuevas muestras.

⚙️ Requisitos

Este proyecto utiliza Python 3.x y las siguientes bibliotecas:

      pip install numpy matplotlib scikit-learn

📊 Resultados

Métricas de pérdida y exactitud durante el entrenamiento 
Reporte de clasificación 
Matriz de Confusión

📈 Visualizaciones

El script genera:

Gráficas de accuracy y loss en entrenamiento y validación.

Matriz de confusión en el conjunto de prueba.

📦 Función de Predicción

Se incluye una función predict para evaluar nuevas muestras (como en el tutorial de Geeks for Geeks)

         X_new = X_test[:10]
         y_pred_new, y_prob_new = predict(X_new, W1, b1, W2, b2)
         print(y_pred_new)  # Clases predichas
         print(y_prob_new)  # Probabilidades


