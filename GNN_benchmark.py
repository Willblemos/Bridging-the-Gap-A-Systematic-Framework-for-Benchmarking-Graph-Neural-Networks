
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.utils import to_dense_adj
from torch.nn import Linear

from IPython.display import HTML
from matplotlib import animation
from torch_geometric.datasets import GNNBenchmarkDataset
from sklearn.manifold import TSNE  # For dimensionality reduction
import time


# Variável global para armazenar os resultados
results_df = pd.DataFrame(columns=['Dataset', 'Nodes', 'Edges', 'Model', 'Num Layers', 'Num Neurons', 'Learning Rate', 'Dropout', 'Train Accuracy', 'Test Accuracy','Execution Time', 'Setup'])

# Load DATASETS ##########################################################
KarateClub_dataset = KarateClub()
Cora_dataset = Planetoid(root='data/Cora', name='Cora')
Citeseer_dataset = Planetoid(root='data/Planetoid', name='Citeseer')
Pubmed_dataset = Planetoid(root='data/Planetoid', name='Pubmed')
# Synthetic_pattern_dataset = GNNBenchmarkDataset(root='data/GNNBenchmark', name='PATTERN', split='train')
# Synthetic_cluster_dataset = GNNBenchmarkDataset(root='data/GNNBenchmark', name='CLUSTER', split='train')

# datasets = [Cora_dataset,Citeseer_dataset,Pubmed_dataset,Synthetic_pattern_dataset,Synthetic_cluster_dataset]
# datasets_names = ['Cora', 'CiteSeer', 'Pubmed', 'Syntethic_pattern', 'Synthetic_cluster']

datasets = [Cora_dataset,Citeseer_dataset,Pubmed_dataset]
datasets_names = ['Cora', 'CiteSeer', 'Pubmed']

for graph in datasets:
  dataset_name = datasets_names[datasets.index(graph)]
  dataset = graph


  # Parâmetros gerais
  num_layers_values = [1,2, 3]
  num_neurons_values = [16, 32, 64]
  num_epochs = 150
  learning_rate_values = [0.01, 0.001, 0.0001]
  dropout_values = [0.5, 0.2, 0.1]  # Variação do dropout

  # Recuperando informações do conjunto de dados
  num_features = dataset.num_features
  num_classes = dataset.num_classes
  num_nodes = dataset.data.num_nodes
  num_edges = dataset.data.num_edges

  # Separando os dados em treino e teste
  data = dataset[0]
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = data.to(device)
  x, edge_index, y = data.x, data.edge_index, data.y

  # Definindo o modelo GCN
  class GCN(torch.nn.Module):
      def __init__(self, num_features, num_classes, num_layers, num_neurons, activation, dropout):
          super(GCN, self).__init__()
          self.convs = torch.nn.ModuleList()
          self.convs.append(GCNConv(num_features, num_neurons))
          for _ in range(num_layers - 1):
              self.convs.append(GCNConv(num_neurons, num_neurons))
          self.output = GCNConv(num_neurons, num_classes)
          self.activation = activation
          self.dropout = dropout

      def forward(self, x, edge_index):
          for conv in self.convs:
              x = conv(x, edge_index)
              x = self.activation(x)
              x = F.dropout(x, p=self.dropout, training=self.training)
          x = self.output(x, edge_index)
          return F.log_softmax(x, dim=1)

  # Definindo o modelo GraphSAGE
  class GraphSAGE(torch.nn.Module):
      def __init__(self, num_features, num_classes, num_layers, num_neurons, activation, dropout):
          super(GraphSAGE, self).__init__()
          self.convs = torch.nn.ModuleList()
          self.convs.append(SAGEConv(num_features, num_neurons, normalize=False))
          for _ in range(num_layers - 1):
              self.convs.append(SAGEConv(num_neurons, num_neurons, normalize=False))
          self.output = SAGEConv(num_neurons, num_classes, normalize=False)
          self.activation = activation
          self.dropout = dropout

      def forward(self, x, edge_index):
          for conv in self.convs:
              x = conv(x, edge_index)
              x = self.activation(x)
              x = F.dropout(x, p=self.dropout, training=self.training)
          x = self.output(x, edge_index)
          return F.log_softmax(x, dim=1)

  # Definindo o modelo GAT
  class GAT(torch.nn.Module):
      def __init__(self, num_features, num_classes, num_layers, num_neurons, activation, dropout):
          super(GAT, self).__init__()
          self.convs = torch.nn.ModuleList()
          self.convs.append(GATConv(num_features, num_neurons))
          for _ in range(num_layers - 1):
              self.convs.append(GATConv(num_neurons, num_neurons))
          self.output = GATConv(num_neurons, num_classes)
          self.activation = activation
          self.dropout = dropout

      def forward(self, x, edge_index):
          for conv in self.convs:
              x = conv(x, edge_index)
              x = self.activation(x)
              x = F.dropout(x, p=self.dropout, training=self.training)
          x = self.output(x, edge_index)
          return F.log_softmax(x, dim=1)

  # Função para criar e treinar um modelo
  def train_model(model, optimizer, criterion, x, edge_index, y, data, num_epochs, model_name, num_layers, num_neurons, learning_rate, dropout):
      print(f'\nTraining {model_name} model...')
      print(f'Number of layers: {num_layers}, Number of neurons: {num_neurons}, Epochs: {num_epochs}, Learning Rate: {learning_rate}, Dropout: {dropout}')

      train_losses, test_losses = [], []
      train_accuracies, test_accuracies = [], []

      for epoch in range(num_epochs):
          model.train()
          optimizer.zero_grad()
          out = model(x, edge_index)
          loss = criterion(out[data.train_mask], y[data.train_mask])
          loss.backward()
          optimizer.step()
          train_losses.append(loss.item())

          model.eval()
          with torch.no_grad():
              out = model(x, edge_index)
              test_loss = criterion(out[data.test_mask], y[data.test_mask])
              test_losses.append(test_loss.item())

              _, predicted = torch.max(out[data.test_mask], 1)
              total = data.test_mask.sum().item()
              correct = predicted.eq(y[data.test_mask]).sum().item()
              test_accuracy = correct / total
              test_accuracies.append(test_accuracy)

          _, predicted = torch.max(out[data.train_mask], 1)
          total = data.train_mask.sum().item()
          correct = predicted.eq(y[data.train_mask]).sum().item()
          train_accuracy = correct / total
          train_accuracies.append(train_accuracy)
          print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Training Loss: {train_losses[-1]:.4f}, '
                f'Test Loss: {test_losses[-1]:.4f}, '
                f'Training Accuracy: {train_accuracies[-1]:.4f}, '
                f'Test Accuracy: {test_accuracies[-1]:.4f}')


      return train_losses, test_losses, train_accuracies, test_accuracies

  # Função para criar o modelo GCN
  def create_gcn_model(num_features, num_classes, num_layers, num_neurons, activation, dropout):
      model = GCN(num_features, num_classes, num_layers, num_neurons, activation, dropout).to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      criterion = torch.nn.NLLLoss()
      return model, optimizer, criterion

  # Função para criar o modelo GraphSAGE
  def create_graphsage_model(num_features, num_classes, num_layers, num_neurons, activation, dropout):
      model = GraphSAGE(num_features, num_classes, num_layers, num_neurons, activation, dropout).to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      criterion = torch.nn.NLLLoss()
      return model, optimizer, criterion

  # Função para criar o modelo GAT
  def create_gat_model(num_features, num_classes, num_layers, num_neurons, activation, dropout):
      model = GAT(num_features, num_classes, num_layers, num_neurons, activation, dropout).to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      criterion = torch.nn.NLLLoss()
      return model, optimizer, criterion

  # Função para armazenar os resultados no dataframe global
  def store_results(dataset_name, num_nodes, num_edges, model_name, num_layers, num_neurons, learning_rate, dropout,
                    train_accuracy, test_accuracy, execution_time, setup):
      global results_df
      results_df = results_df.append({
          'Dataset': dataset_name,
          'Nodes': num_nodes,
          'Edges': num_edges,
          'Model': model_name,
          'Num Layers': num_layers,
          'Num Neurons': num_neurons,
          'Learning Rate': learning_rate,
          'Dropout': dropout,
          'Train Accuracy': train_accuracy,
          'Test Accuracy': test_accuracy,
          'Execution Time': execution_time,
          'Setup': setup
      }, ignore_index=True)

  # Loop para executar o experimento com diferentes hiperparâmetros
  experiment_num = 1

  for num_layers in num_layers_values:
      for num_neurons in num_neurons_values:
          for learning_rate in learning_rate_values:
              for dropout in dropout_values:  # Variação do dropout
                  start_time = time.time()
                  # Criando e treinando o modelo GCN
                  gcn_model, gcn_optimizer, gcn_criterion = create_gcn_model(num_features, num_classes, num_layers, num_neurons, F.relu, dropout)  # Dropout passado para a criação do modelo
                  gcn_train_losses, gcn_test_losses, gcn_train_accuracies, gcn_test_accuracies = train_model(
                      gcn_model, gcn_optimizer, gcn_criterion, x, edge_index, y, data, num_epochs, 'GCN', num_layers, num_neurons, learning_rate, dropout
                  )
                  end_time = time.time()
                  gcn_execution_time = end_time - start_time
                  store_results(dataset_name, num_nodes, num_edges, 'GCN', num_layers, num_neurons, learning_rate, dropout,
                                gcn_train_accuracies[-1], gcn_test_accuracies[-1], gcn_execution_time ,experiment_num)

                  # Criando e treinando o modelo GraphSAGE
                  start_time = time.time()
                  graphsage_model, graphsage_optimizer, graphsage_criterion = create_graphsage_model(
                      num_features, num_classes, num_layers, num_neurons, F.relu, dropout
                  )
                  graphsage_train_losses, graphsage_test_losses, graphsage_train_accuracies, graphsage_test_accuracies = train_model(
                      graphsage_model, graphsage_optimizer, graphsage_criterion, x, edge_index, y, data, num_epochs, 'GraphSAGE', num_layers, num_neurons, learning_rate, dropout
                  )
                  end_time = time.time()
                  sage_execution_time = end_time - start_time
                  store_results(dataset_name, num_nodes, num_edges, 'GraphSAGE', num_layers, num_neurons, learning_rate, dropout,
                                graphsage_train_accuracies[-1], graphsage_test_accuracies[-1], sage_execution_time, experiment_num)

                  # Criando e treinando o modelo GAT
                  start_time = time.time()
                  gat_model, gat_optimizer, gat_criterion = create_gat_model(num_features, num_classes, num_layers, num_neurons, F.elu, dropout)
                  gat_train_losses, gat_test_losses, gat_train_accuracies, gat_test_accuracies = train_model(
                      gat_model, gat_optimizer, gat_criterion, x, edge_index, y, data, num_epochs, 'GAT', num_layers, num_neurons, learning_rate, dropout
                  )
                  end_time = time.time()
                  gat_execution_time = end_time - start_time
                  store_results(dataset_name, num_nodes, num_edges, 'GAT', num_layers, num_neurons, learning_rate, dropout,
                                gat_train_accuracies[-1], gat_test_accuracies[-1], gat_execution_time,experiment_num)

                  experiment_num += 1

# Salvar os resultados em um arquivo CSV
results_df.to_csv('results.csv', index=False)
