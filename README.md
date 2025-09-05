# Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
te proyecto implementa una red neuronal feedforward para clasificación binaria, programada desde cero en Python utilizando únicamente NumPy y matplotlib. El objetivo es comprender el funcionamiento interno de una red neuronal, sin depender de frameworks de alto nivel como TensorFlow o PyTorch.

📌 Características principales

Generación de un dataset sintético balanceado con scikit-learn.
Implementación manual de:
    *Forward Propagation
    
    Función de pérdida (Cross-Entropy)
    
    Backpropagation (regla de la cadena, derivadas parciales)
    
    Entrenamiento con ajuste de pesos y sesgos mediante Gradient Descent.
    
    Evaluación con accuracy, loss, classification report y matriz de confusión.
    
    Gráficas de curvas de aprendizaje (train vs validation).
    
    Función de predicción para nuevas muestras.
