import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#Dataset

X, y = make_classification(
    n_samples=7000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=42
)

#Dividir  en 60% train, 20% val, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=43, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=43, stratify=y_temp
)
np.random.seed(0)

#Función de activación de sigmoide
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def inicializacion(n_input=2, n_hidden=3, n_output=1):
    #Pesos de input a hidden
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.ones(n_hidden)

    #Pesos de hidden a output
    W2 = np.random.randn(n_hidden, n_output)
    b2 = np.ones(n_output)
    return W1, b1, W2, b2

def fprop(X, W1, b1, W2, b2): 
    z1 = X.dot(W1) + b1   #De input a la capa intermedia 
    act1 = sigmoid(z1)    # activation function 1

    z2 = act1.dot(W2) + b2    #De capa intermedia a capa de salida
    act2 = sigmoid(z2)        #salida binaria --> y_hat (predicha)

    return z1, act1, z2, act2   

#Usar cross enthropy para la función de pérdida 
def loss(y, y_hat):
    N = y.shape[0]  #número de muestras

    #Evitar log(0) y aplanar para que sea 1D
    y_hat = np.clip(y_hat.flatten(), 1e-10, 1-1e-10)

    #Calcular la suma de todos los errores
    total_loss = 0.0
    for i in range(N):
        p_i = y_hat[i]   
        total_loss += -(y[i]*np.log(p_i) + (1-y[i])*np.log(1-p_i))

    return total_loss / N



def bprop(X, y, z1, act1, act2, W1, b1, W2, b2, alpha=0.02):
    #Error en la capa de salida
    d2 = (act2 - y.reshape(-1,1))

    #Error en la capa oculta
    d1 = (d2.dot(W2.T)) * (act1*(1-act1))

    #Gradiente para los pesos y los sesgos
    nabla_W2 = act1.T.dot(d2)
    nabla_b2 = np.mean(d2, axis=0)
   
    nabla_W1 = X.T.dot(d1)
    nabla_b1 = np.mean(d1, axis=0)

     #Actualización de los pesos y sesgos
    W1 -= alpha * nabla_W1
    b1 -= alpha * nabla_b1
    W2 -= alpha * nabla_W2
    b2 -= alpha * nabla_b2

    return W1, b1, W2, b2

def accuracy(y, y_hat):
    y_hat_labels = (y_hat > 0.5).astype(int).flatten()
    return np.mean(y == y_hat_labels)

#Entrenamiento
def train(X_train, y_train, X_val, y_val, W1, b1, W2, b2, alpha=0.02, epochs=300):
    acc_train, loss_train = [], []
    acc_val, loss_val = [], []

    for j in range(epochs):
        #Forward con el training set
        z1, act1, z2, act2 = fprop(X_train, W1, b1, W2, b2)
        L_train = loss(y_train, act2)
        acc_train.append(accuracy(y_train, act2))
        loss_train.append(L_train)

        #Forward  con el validation set
        _, _, _, act2_val = fprop(X_val, W1, b1, W2, b2)
        L_val = loss(y_val, act2_val)
        acc_val.append(accuracy(y_val, act2_val))
        loss_val.append(L_val)

        #Backprop (solo en training set)
        W1, b1, W2, b2 = bprop(X_train, y_train, z1, act1, act2, W1, b1, W2, b2, alpha)

        #Cada 10 epochs, imprime las métricas de loss y accuracy para el training set, y loss y accuracy para el validation set
        if (j+1) % 10 == 0:
            print(f"Epoch {j+1}/{epochs} - Train Loss: {L_train:.4f}, Train Acc: {acc_train[-1]:.3f}, Val Loss: {L_val:.4f}, Val Acc: {acc_val[-1]:.3f}")

    return acc_train, loss_train, acc_val, loss_val, W1, b1, W2, b2


#Entrenar la red (train())
W1, b1, W2, b2 = inicializacion()
acc_train, loss_train, acc_val, loss_val, W1, b1, W2, b2 = train(X_train, y_train, X_val, y_val, W1, b1, W2, b2, alpha=0.02, epochs=100)


#Graficar el training set junto con el validation set para ver si hay overfitting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

ax1.plot(acc_train, label="Train")
ax1.plot(acc_val, label="Validation")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(loss_train, label="Train")
ax2.plot(loss_val, label="Validation")
ax2.set_title("Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.show()


#Evaluar con el testing set
_, _, _, y_pred_test = fprop(X_test, W1, b1, W2, b2)
y_pred_labels = (y_pred_test > 0.55).astype(int).flatten()

#Reporte clasificación 
print("\n=== Clasification Report en Test ===")
print(classification_report(y_test, y_pred_labels))

#Matriz de confusión
cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.show()



def predict(X_new, W1, b1, W2, b2, threshold=0.5):
    #Usar la función de forward para obtener la salida
    _, _, _, y_prob = fprop(X_new, W1, b1, W2, b2)

    #Calcular probabilidades en clases según el threshold
    y_pred = []
    for p in y_prob.flatten():
        if p > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return np.array(y_pred), y_prob.flatten()


#Nada más se usan los primeros 10 datos del training set.
X_new = X_test[:10]
y_pred_new, y_prob_new = predict(X_new, W1, b1, W2, b2)

print("Probabilidades:", np.round(y_prob_new, 3))
print("Clases predichas:", y_pred_new)
print("Clases reales:  ", y_test[:10])

