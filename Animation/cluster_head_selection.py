import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

# Create a new graph
G = nx.Graph()

# Add the central node
central_node = 0
G.add_node(central_node)

# Add surrounding nodes and connect them to the central node
surrounding_nodes = list(range(1, 6))  # Nodes 1 to 5
for node in surrounding_nodes:
    G.add_node(node)
    G.add_edge(central_node, node)

# Define positions for each node in a circular layout around the center
pos = {central_node: (0, 0)}  # Central node at (0, 0)
radius = 1.5  # Distance from the central node to surrounding nodes

# Calculate positions for surrounding nodes in a circular pattern
for i, node in enumerate(surrounding_nodes):
    angle = 2 * np.pi * i / len(surrounding_nodes)  # Divide the circle
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    pos[node] = (x, y)

# Assign random energy levels between 0.1 and 0.9 to surrounding nodes
energy_levels = {node: round(random.uniform(0.1, 0.9), 2) for node in surrounding_nodes}

# Calculate the average energy of surrounding nodes
surrounding_energy_avg = sum(energy_levels[node] for node in surrounding_nodes) / len(surrounding_nodes)

# Set the central node's energy to be less than half of the surrounding nodes' average energy
energy_levels[central_node] = round(surrounding_energy_avg / 2 - 0.05, 2)  # Slightly less than half

# Set colors for nodes, with central node colored differently if its energy is lower
central_color = "red" if energy_levels[central_node] < (surrounding_energy_avg / 2) else "skyblue"
node_colors = [central_color if node == central_node else "skyblue" for node in G.nodes()]

# Draw the graph
fig, ax = plt.subplots(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, ax=ax)

# Add energy level labels for each node
for node, (x, y) in pos.items():
    energy = energy_levels[node]
    ax.text(x, y + 0.15, f"Energy: {energy}", horizontalalignment="center", fontsize=10, color="blue")

# plt.title("Central Node with Surrounding Nodes and Energy Levels")
plt.savefig('cluster_head_selection.png')
plt.show()
