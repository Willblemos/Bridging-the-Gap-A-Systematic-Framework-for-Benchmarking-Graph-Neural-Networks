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
import pandas as pd


# Load DATASETS ##########################################################
KarateClub_dataset = KarateClub()
Cora_dataset = Planetoid(root='data/Cora', name='Cora')
Citeseer_dataset = Planetoid(root='data/Planetoid', name='Citeseer')
Pubmed_dataset = Planetoid(root='data/Planetoid', name='Pubmed')
Synthetic_pattern_dataset = GNNBenchmarkDataset(root='data/GNNBenchmark', name='PATTERN', split='train')
Synthetic_cluster_dataset = GNNBenchmarkDataset(root='data/GNNBenchmark', name='CLUSTER', split='train')
# Create an empty list to store the results for each model
results = []
metrics = []

#dataset = Pubmed_dataset

#datasets = [Cora_dataset,Citeseer_dataset,Synthetic_pattern_dataset,Synthetic_cluster_dataset]
#datasets = [Cora_dataset,Citeseer_dataset]
datasets = [Synthetic_pattern_dataset,Synthetic_cluster_dataset]


for dataset in datasets:

  # Print information about the dataset
  print('------------------------------------------------------------------------------------')
  print(f"Information about Dataset: {dataset}")
  print('------------')
  print(f'Number of graphs: {len(dataset)}')
  print(f'Number of features: {dataset.num_features}')
  print(f'Number of classes: {dataset.num_classes}')
  print(f'Number of nodes (vertices): {dataset[0].num_nodes}')
  print(f'Number of edges: {dataset[0].num_edges}')
  print('------------------------------------------------------------------------------------')

  data = dataset[0]
  A = to_dense_adj(data.edge_index)[0].numpy().astype(int)

  # Set hyperparameters
  num_features = dataset.num_features
  num_classes = dataset.num_classes
  lr = 0.0001
  num_epochs = 50
  num_hidden_channels = 8
  num_layers = 1

  G = to_networkx(data, to_undirected=True)
  reconstructed_embeddings = []

  # Define GCN model
  class GCN(torch.nn.Module):
      def __init__(self, num_features, hidden_channels, num_classes, num_layers):
          super(GCN, self).__init__()
          self.convs = torch.nn.ModuleList()
          self.convs.append(GCNConv(num_features, hidden_channels))
          for i in range(num_layers - 1):
              self.convs.append(GCNConv(hidden_channels, hidden_channels))
          self.out = Linear(hidden_channels, num_classes)

      def forward(self, x, edge_index):
          for conv in self.convs:
              x = conv(x, edge_index).relu()
          z = self.out(x)
          return x, z

  # Define GraphSAGE model
  class GraphSAGE(torch.nn.Module):
      def __init__(self, num_features, hidden_channels, num_classes, num_layers):
          super(GraphSAGE, self).__init__()
          self.convs = torch.nn.ModuleList()
          self.convs.append(SAGEConv(num_features, hidden_channels))
          for i in range(num_layers - 1):
              self.convs.append(SAGEConv(hidden_channels, hidden_channels))
          self.out = Linear(hidden_channels, num_classes)

      def forward(self, x, edge_index):
          for conv in self.convs:
              x = conv(x, edge_index).relu()
          z = self.out(x)
          return x, z

  # Define GAT model
  class GAT(torch.nn.Module):
      def __init__(self, num_features, hidden_channels, num_classes, num_layers):
          super(GAT, self).__init__()
          self.convs = torch.nn.ModuleList()
          self.convs.append(GATConv(num_features, hidden_channels))
          for i in range(num_layers - 1):
              self.convs.append(GATConv(hidden_channels, hidden_channels))
          self.out = Linear(hidden_channels, num_classes)

      def forward(self, x, edge_index):
          for conv in self.convs:
              x = conv(x, edge_index).relu()
          z = self.out(x)
          return x, z

  # Function to calculate accuracy
  def accuracy(pred_y, y):
      return (pred_y == y).sum() / len(y)



  # Training loop function
  def train_model(model, data, criterion, optimizer, num_epochs=201):
      embeddings = []
      losses = []
      accuracies = []
      outputs = []

      for epoch in range(num_epochs):
          # Clear gradients
          optimizer.zero_grad()

          # Forward pass
          h, z = model(data.x, data.edge_index)

          # Calculate loss function
          loss = criterion(z, data.y)

          # Calculate accuracy
          acc = accuracy(z.argmax(dim=1), data.y)

          # Compute gradients
          loss.backward()

          # Tune parameters
          optimizer.step()

          # Store data for animations
          embeddings.append(h)
          losses.append(loss.item())
          accuracies.append(acc.item())
          outputs.append(z.argmax(dim=1))

          # Print metrics every 10 epochs
          if epoch % 10 == 0:
              print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

      return embeddings, losses, accuracies, outputs


  # Function to train and evaluate a model
  def train_and_evaluate_model(model, model_name, data, criterion, optimizer, num_epochs=201):
      print(f"\nTraining {model_name}...")
      print(f"Dataset: {dataset}")
      print(f"Number of Features: {num_features}, Hidden Channels: {num_hidden_channels}, Number of Classes: {num_classes}, Learning Rate: {lr}, Number of Epochs: {num_epochs}, Number of Layers: {num_layers}")

      start_time = time.time()
      mem_usage = []

      embeddings, losses, accuracies, outputs = train_model(model, data, criterion, optimizer, num_epochs=num_epochs)
      end_time = time.time()
      execution_time = end_time - start_time

      # Calculate assortativity coefficient based on the last graph embedding
      final_h = embeddings[-1]
      #final_h_np = final_h.detach().cpu().numpy()
      #final_G = nx.Graph(nx.from_numpy_array(np.dot(final_h_np, final_h_np.T)))
      # original_metrics = Graph_metrics(G)
      # embedded_metrics = Graph_metrics(final_G)
      reconstructed_embeddings.append(final_h)

      # Print metrics
      print(f"\nMetrics for {model_name}:")
      print(f"Final Loss: {losses[-1]:.4f}")
      print(f"Final Accuracy: {accuracies[-1] * 100:.2f}%")
      print(f"Execution Time: {execution_time:.2f} seconds")
      #return losses, accuracies, outputs, original_metrics, embedded_metrics, execution_time
      return losses, accuracies, outputs, execution_time, final_h


  # Initialize models and optimizer
  gcn_model = GCN(num_features, num_hidden_channels, num_classes, num_layers)
  graphsage_model = GraphSAGE(num_features, num_hidden_channels, num_classes, num_layers)
  gat_model = GAT(num_features, num_hidden_channels, num_classes, num_layers)

  models = [gcn_model, graphsage_model, gat_model]
  model_names = ['GCN', 'GraphSAGE', 'GAT']

  criterion = torch.nn.CrossEntropyLoss()

  # Train and evaluate models
  for model, model_name in zip(models, model_names):
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      #losses, accuracies, outputs, original_metrics, embedded_metrics,execution_time = train_and_evaluate_model(model, model_name, data, criterion, optimizer, num_epochs=num_epochs)
      losses, accuracies, outputs, execution_time, embeddings = train_and_evaluate_model(model, model_name, data, criterion, optimizer, num_epochs=num_epochs)


      # Add the model results to the list
      results.append({
          "Model": model_name,
          "Dataset": str(dataset),
          "Number of Features": num_features,
          "Hidden Channels": num_hidden_channels,
          "Number of Classes": num_classes,
          "Learning Rate": lr,
          "Number of Epochs": num_epochs,
          "Number of Layers": num_layers,
          "Final Loss": losses[-1],
          "Final Accuracy": accuracies[-1] * 100,
          "Execution Time (seconds)": execution_time,
      })

      metrics.append([str(dataset),model_name, embeddings])

# Create a DataFrame from the results list
embedding_results = pd.DataFrame(results)

def reconstruct_graph(embedded_graph):
  final_h_np = embedded_graph.detach().cpu().numpy()
  reconstructed_graph = nx.Graph(nx.from_numpy_array(np.dot(final_h_np, final_h_np.T)))
  return (reconstructed_graph)


def Graph_metrics(embedding_output):
    print("***** Running metrics for: " + str(embedding_output[0]))
    graph = reconstruct_graph(embedding_output[2])
    assortativity_coefficient = nx.degree_assortativity_coefficient(graph)
    global_efficiency = nx.global_efficiency(graph)
    transitivity = nx.transitivity(graph)
    triadic_census = 0
    motif_counts = 0
    clustering_coefficient = 0

    return [
        embedding_output[0],
        embedding_output[1],
        assortativity_coefficient,
        global_efficiency,
        #triadic_census,
        #motif_counts,
        #clustering_coefficient,
        transitivity
    ]

network_results = []
for model in metrics:
  print("-----------------------------------------------------------------------------")


  dataset,model_name,assortativity_coefficient,global_efficiency,transitivity = Graph_metrics(model)
  network_results.append({
    "Dataset": str(dataset),
    "Model": model_name,
    "assortativity_coefficient": assortativity_coefficient,
    "global_efficiency": global_efficiency,
    "transitivity": transitivity,
})
  print("-----------------------------------------------------------------------------")
network_analysis = pd.DataFrame(network_results)
print(network_analysis)