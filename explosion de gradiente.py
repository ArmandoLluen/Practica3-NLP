# Importación de librerías necesarias
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Paso 1: Preparación de Datos
# Leer y tokenizar el texto desde el archivo de entrada
with open('Ejemplo.txt', 'r') as file:
    text = file.read()

# Crear vocabulario de caracteres únicos
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convertir texto a índices de caracteres
encoded_text = [char_to_idx[ch] for ch in text]

# Parámetros de entrada
sequence_length = 200  # Aumentar longitud de secuencia
num_layers = 6  # Aumentar el número de capas
vocab_size = len(chars)
batch_size = 256


# Definir dataset para secuencias de caracteres
class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_len]),
                torch.tensor(self.data[idx + 1:idx + self.seq_len + 1]))


# Crear el dataloader
dataset = TextDataset(encoded_text, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Paso 2: Definición del Modelo RNN
class CompleteRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_capas):
        super(CompleteRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_capas  # Número de capas
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_capas, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()  # Añadir capa tanh

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.tanh(out)  # Aplicar tanh en la salida
        out = self.fc(out.reshape(-1, self.hidden_dim))  # Redimensionar para la salida
        return out, hidden

    def init_hidden(self, batch_size):
        # Estado oculto inicial en ceros, con num_layers capas
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


# Paso 3: # Función de Entrenamiento con ajuste dinámico del tamaño del estado oculto
def train(modelo, dataloader, criterion, optimizer, epocas):
    modelo.train()
    grad_norms = []  # Para guardar la norma de los gradientes en cada batch

    # Bucle principal de entrenamiento
    for epoch in range(epocas):
        total_loss = 0

        # Entrenamiento por lotes
        for inputs, targets in dataloader:
            # Ajustar el tamaño del estado oculto al tamaño actual del lote
            batch_size = inputs.size(0)
            hidden = modelo.init_hidden(batch_size).detach()

            optimizer.zero_grad()

            # Cálculo de la pérdida (forward) y retropropagación (backward)
            outputs, hidden = modelo(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            # Calcular la norma de los gradientes y guardarlos
            grad_norm = 0
            for p in modelo.rnn.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()
            grad_norms.append(grad_norm)

            # Actualización de los parámetros del modelo
            optimizer.step()
            total_loss += loss.item()

        print(f'Época [{epoch + 1}/{epocas}], Pérdida: {total_loss / len(dataloader):.4f}')

    # Paso 4: Visualización de la Norma de los Gradientes
    plt.plot(grad_norms)
    plt.xlabel('Batch')
    plt.ylabel('Norma del Gradiente')
    plt.title('Norma de los Gradientes Durante el Entrenamiento')
    plt.show()


# Paso 5: Configuración del Modelo y Entrenamiento
# Parámetros del modelo
embedding_dim = 64
hidden_dim = 128
output_dim = vocab_size
learning_rate = 0.0001  # Reducir tasa de aprendizaje
epochs = 25

# Inicializar modelo, criterio y optimizador
model = CompleteRNN(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenar el modelo y visualizar la norma de los gradientes
train(model, dataloader, criterion, optimizer, epochs)

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mover el modelo al dispositivo
model.to(device)


# Calcular las metricas de perdida de validacion, acurracy y perplejidad
def evaluate_model(modelo, dataloader, criterion):
    modelo.eval()  # Modo evaluación
    total_loss = 0
    total_correct = 0
    total_chars = 0

    with torch.no_grad():  # Desactivar gradientes para evaluación
        for inputs, targets in dataloader:
            # Mover los datos al dispositivo (GPU o CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Ajuste del tamaño del lote
            batch_size = inputs.size(0)
            hidden = modelo.init_hidden(batch_size).to(device)

            # Forward
            outputs, hidden = modelo(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            total_loss += loss.item() * inputs.size(0)

            # Calcular precisión
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets.view(-1)).sum().item()
            total_chars += targets.numel()

    # Cálculo de la pérdida promedio y la perplejidad
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = math.exp(avg_loss)

    # Cálculo de la precisión
    accuracy = total_correct / total_chars

    return avg_loss, perplexity, accuracy


# Evaluacion de la generacion de texto
def generate_text(modelo, start_str, length=200):
    modelo.eval()
    caracteres = [char_to_idx[c] for c in start_str]
    input_seq = torch.tensor(caracteres).unsqueeze(0).to(device)
    hidden = modelo.init_hidden(1).to(device)

    texto_generado = start_str

    with torch.no_grad():
        for _ in range(length):
            output, hidden = modelo(input_seq, hidden)
            prob = torch.nn.functional.softmax(output[-1], dim=0)
            char_idx = torch.multinomial(prob, 1).item()
            texto_generado += idx_to_char[char_idx]
            input_seq = torch.tensor([[char_idx]]).to(device)

    print("\nTexto Generado:")
    print(texto_generado)
    return texto_generado


# Paso 6: Evaluación

avg_loss, perplexity, accuracy = evaluate_model(model, dataloader, criterion)
print(f'Pérdida Promedio en Validación: {avg_loss:.4f}')
print(f'Perplejidad: {perplexity:.4f}')
print(f'Exactitud (Accuracy): {accuracy * 100:.2f}%')

start_str = "King"
generated_text = generate_text(model, start_str, length=200)
