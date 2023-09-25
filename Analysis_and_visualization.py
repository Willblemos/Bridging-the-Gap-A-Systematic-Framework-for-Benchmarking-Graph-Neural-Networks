data = results_df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data, columns=[
    "Dataset", "Nodes", "Edges", "Model", "Num Layers", "Num Neurons", "Learning Rate",
    "Dropout", "Train Accuracy", "Test Accuracy", "Setup"
])

# Set up the plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Relationship between Learning Rate and Test Accuracy
plt.subplot(2, 2, 1)
sns.scatterplot(x="Learning Rate", y="Test Accuracy", data=df, hue="Model", legend="full")
plt.title("Learning Rate vs. Test Accuracy")

# Relationship between Dropout and Test Accuracy
plt.subplot(2, 2, 2)
sns.scatterplot(x="Dropout", y="Test Accuracy", data=df, hue="Model", legend="full")
plt.title("Dropout vs. Test Accuracy")

# Performance comparison among different GNN models
plt.subplot(2, 2, 3)
sns.boxplot(x="Model", y="Test Accuracy", data=df)
plt.title("Performance Comparison among GNN Models")

# Relationship between the number of neurons and Test Accuracy
plt.subplot(2, 2, 4)
sns.scatterplot(x="Num Neurons", y="Test Accuracy", data=df, hue="Model", legend="full")
plt.title("Number of Neurons vs. Test Accuracy")

plt.tight_layout()
plt.show()

##########################
import pandas as pd
import matplotlib.pyplot as plt

# Calcular a acurácia média de teste para cada setup
mean_test_accuracy = data.groupby('Setup')['Test Accuracy'].mean()

# Ordenar os índices pela acurácia média de teste em ordem crescente
mean_test_accuracy_sorted = mean_test_accuracy.sort_values()

# Plotar os resultados ordenados
plt.figure(figsize=(20, 6))
mean_test_accuracy_sorted.plot(kind='bar', color='blue')
plt.title('Average Accuracy by setup')
plt.xlabel('Setup')
plt.ylabel('Avg test accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##########################
import matplotlib.pyplot as plt
import seaborn as sns

# Lista de modelos únicos
models = data['Model'].unique()

# Definir a paleta de cores
palette = sns.color_palette("husl", n_colors=len(models))

# Lista de hiperparâmetros
hyperparams = ['Num Layers', 'Num Neurons', 'Learning Rate', 'Dropout']

# Loop pelos hiperparâmetros
for param in hyperparams:
    plt.figure(figsize=(8, 6))

    for i, model in enumerate(models):
        # Filtrar dados para o modelo atual
        model_data = data[data['Model'] == model]

        # Agrupar os dados por hiperparâmetro e calcular as médias das métricas
        hyperparams_grouped_data = model_data.groupby(param).mean()

        # Cor baseada na paleta de cores e no índice do modelo
        color = palette[i]

        # Plotar linha de sensibilidade para o modelo atual
        sns.lineplot(data=hyperparams_grouped_data.reset_index(), x=param, y='Test Accuracy', color=color, label=model)

    plt.xlabel(param)
    plt.ylabel('Avg Test accuracy')
    plt.title(f'Accuracy sensibility to {param}')
    plt.legend()
    plt.grid(True)

    # Definir os tickmarks com base nos valores reais presentes nos dados
    plt.xticks(hyperparams_grouped_data.index)

    plt.show()

    ###########################


# Criar um DataFrame para armazenar os resultados
summary_data = []

# Para cada modelo, calcular as métricas e adicionar ao DataFrame
for model in data['Model'].unique():
    model_data = data[data['Model'] == model]
    datasets = model_data['Dataset'].unique()
    for dataset in datasets:
        dataset_data = model_data[model_data['Dataset'] == dataset]
        model_summary = {
            'Model': model,
            'Dataset': dataset,
            'Min Test Accuracy': dataset_data['Test Accuracy'].min(),
            'Max Test Accuracy': dataset_data['Test Accuracy'].max(),
            'Mean Test Accuracy': dataset_data['Test Accuracy'].mean(),
            'Standard Deviation': dataset_data['Test Accuracy'].std()
        }
        summary_data.append(model_summary)

# Criar o DataFrame final com os resultados
summary_df = pd.DataFrame(summary_data)

def plot_accuracy_comparison(data, model_name):
    # Filtrar os dados apenas para o modelo especificado
    model_data = data[data['Model'] == model_name]

    # Agrupar os dados por dataset e calcular as médias das métricas para o modelo especificado
    grouped_data = model_data.groupby('Dataset').mean()

    # Plotar gráfico de barras para as médias de acurácia de treinamento e teste para o modelo especificado
    plt.figure(figsize=(12, 8))
    ax = grouped_data[['Train Accuracy', 'Test Accuracy']].plot(kind='bar')
    plt.title(f'Average Train and Testing accuracy for {model_name} by dataset')
    plt.ylabel('Accuracy')
    plt.xlabel('Dataset')
    plt.xticks(rotation=45)

    # Adicionar os valores acima das barras
    for i, value in enumerate(grouped_data['Train Accuracy']):
        ax.text(i - 0.15, value + 0.01, f'{value:.3f}', color='black', fontsize=10)

    for i, value in enumerate(grouped_data['Test Accuracy']):
        ax.text(i + 0.15, value + 0.01, f'{value:.3f}', color='black', fontsize=10)

    # Reposicionar a legenda fora da imagem
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    plt.tight_layout()
    plt.show()

# Lista dos modelos presentes nos dados
model_list = data['Model'].unique()

# Para cada modelo, plote o gráfico de comparação de acurácia
for model in model_list:
    plot_accuracy_comparison(data, model)