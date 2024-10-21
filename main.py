from sklearn.linear_model import Perceptron
import numpy as np

# Define a seed para garantir consistência nas execuções
np.random.seed(42)

# Definição do RU
RU = 4121047

# Função para dividir um número em dígitos
def split_digits(number):
    return [int(digit) for digit in str(number)]

# Função para gerar dados de treinamento
def generate_training_data(n_samples=300):
    X_train = []
    y_train = []
    
    # Gerar números aleatórios em torno de RU e classificá-los
    for _ in range(n_samples):
        num = np.random.randint(RU - 1e6, RU + 1e6)
        digits = split_digits(num)  # Chama a função que divide o número em dígitos
        X_train.append(digits)  # Adiciona os dígitos como entrada para o treinamento
        y_train.append(1 if num >= RU else -1)  # 1 se maior ou igual ao RU, -1 se menor
    
    return np.array(X_train), np.array(y_train)

# Gerar dados de treinamento
X_train, y_train = generate_training_data()

# Cria o neuronio
neuronio = Perceptron()

# Treinamento do neuronio
print("Iniciando treinamento do Perceptron...\n")
neuronio.fit(X_train, y_train)

# Previsões para o conjunto de treinamento
y_pred_train = neuronio.predict(X_train)

# Cálculo da taxa de acerto
accuracy_train = np.mean(y_pred_train == y_train) * 100
print(f"Treinamento concluído com taxa de acerto de {accuracy_train:.2f}% no conjunto de treinamento.\n")

# Exibir todos os resultados do treinamento
print("Resultados do treinamento:")
for i in range(len(X_train)):
    print(f"Amostra {i+1}: Dígitos {X_train[i]}, Classificação esperada: {y_train[i]}, Previsão: {y_pred_train[i]}")
print("\n")

# Exibindo no console os pesos e o bias do neuronio
print("Pesos finais do Perceptron:")
for i, weight in enumerate(neuronio.coef_[0]):
    print(f"w{i+1} = {weight}")
print(f"Bias = {neuronio.intercept_[0]}\n")

# Solicitação de 7 dígitos do usuário
inputs = []
for i in range(7):
    digit = int(input(f"Entrada x{i + 1}: "))
    inputs.append(digit)

# Convertendo a lista de entradas em um array NumPy
X_test = np.array([inputs])

# Faz a previsão com o neuronio treinado
prediction = neuronio.predict(X_test)

# Mostra o resultado
number_input = int(''.join(map(str, inputs)))  # Convertendo os dígitos em um número
if prediction == 1:
    print(f"O número {number_input} é maior ou igual a {RU}.")
else:
    print(f"O número {number_input} é menor que {RU}.")