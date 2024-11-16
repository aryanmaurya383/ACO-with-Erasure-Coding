# import matplotlib.pyplot as plt
# import networkx as nx
# import matplotlib.animation as animation
# import random

# # Define the graph structure with forward and reverse edges
# G = nx.DiGraph()
# edges = {
#     (0, 1): 5, (0, 2): 8, (0, 3): 12,
#     (1, 4): 6, (1, 5): 9,
#     (2, 4): 6, (2, 5): 9,
#     (3, 4): 6, (3, 5): 9,
#     (4, 6): 6, (5, 6): 3,
#     (1, 0): 5, (2, 0): 8, (3, 0): 12,
#     (4, 1): 6, (5, 1): 9,
#     (4, 2): 6, (5, 2): 9,
#     (6, 4): 6, (6, 5): 3
# }

# # Add nodes and edges to the graph
# for edge, weight in edges.items():
#     G.add_edge(edge[0], edge[1], weight=weight)

# # Define positions for each node
# pos = {
#     0: (0, 0), 1: (1, 1), 2: (1, 0), 3: (1, -1), 
#     4: (2, 0.5), 5: (2, -0.5), 6: (3, 0)
# }

# # Initialize pheromone levels for edges
# pheromone_levels = {edge: 0.1 for edge in G.edges}

# # Initialize energy levels for nodes (random values between 0.1 and 0.9)
# energy_levels = {node: round(random.uniform(0.1, 0.9), 2) for node in G.nodes()}

# # Ant's starting position
# ant_position = 0

# # Function to draw the graph with updated pheromone levels and energy labels
# def draw_graph():
#     ax.clear()  # Clear the axis for a fresh plot
    
#     # Draw nodes, highlight source and sink with specific colors
#     node_colors = ['slategray' if node == 0 else 'red' if node == 6 else 'lightblue' for node in G.nodes()]
#     nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, ax=ax, arrows=False)
    
#     # Draw edges with thickness based on dynamic pheromone levels
#     for edge, weight in edges.items():
#         nx.draw_networkx_edges(
#             G, pos, edgelist=[edge],
#             width=pheromone_levels[edge] * 2,  # Thickness proportional to pheromone level
#             edge_color="green",
#             ax=ax,
#             arrows=False
#         )
        
#     # Draw edge labels for weights
#     edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    
#     # Draw energy level as a label next to each node
#     for node, (x, y) in pos.items():
#         ax.text(
#             x + 0.1, y + 0.1,  # Offset label position slightly from node
#             f"Energy: {energy_levels[node]}",
#             fontsize=8, color="blue"
#         )

# # Function to update the ant's position
# def update_ant_position(frame):
#     global ant_position
#     draw_graph()  # Redraw the graph
    
#     # Get neighbors of the current node
#     neighbors = list(G.neighbors(ant_position))
#     if neighbors:
#         # Move to a random neighbor
#         ant_position = random.choice(neighbors)
    
#     # Draw the ant as a red circle at the new position
#     ant_x, ant_y = pos[ant_position]
#     ax.plot(ant_x, ant_y, 'ro', markersize=10)  # Draw the ant

# # Set up figure and axis
# fig, ax = plt.subplots(figsize=(8, 5))

# # Create animation
# ani = animation.FuncAnimation(fig, update_ant_position, frames=20, interval=1000)

# plt.show()
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import numpy as np
import random


# Define the graph structure
G = nx.DiGraph()
edges = {
    (0, 1): 5, (0, 2): 8, (0, 3): 12,
    (1, 4): 6, (1, 5): 9,
    (2, 4): 6, (2, 5): 9,
    (3, 4): 6, (3, 5): 9,
    (4, 6): 6, (5, 6): 3,
    (1, 0): 5, (2, 0): 8, (3, 0): 12,
    (4, 1): 6, (5, 1): 9,
    (4, 2): 6, (5, 2): 9,
    (6, 4): 6, (6, 5): 3
}

# Add nodes and edges to the graph
for edge, weight in edges.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# Define positions for each node
pos = {
    0: (0, 0), 1: (1, 1), 2: (1, 0), 3: (1, -1), 
    4: (2, 0.5), 5: (2, -0.5), 6: (3, 0)
}

# Sequence of nodes for the ant to visit
path = [0, 2, 4, 6]
frames_per_edge = 10  # Number of frames for each edge transition
total_frames = frames_per_edge * (len(path) - 1)  # Total frames needed for the entire path
energy_levels = {node: round(random.uniform(0.1, 0.9), 2) for node in G.nodes()}

# Function to draw the graph
def draw_graph():
    ax.clear()
    node_colors = ['slategray' if node == 0 else 'red' if node == 6 else 'lightblue' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, ax=ax, arrows=False)
    
    # Draw edges with weights
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    for node, (x, y) in pos.items():
        ax.text(
            x + 0.1, y + 0.1,  # Offset label position slightly from node
            f"Energy: {energy_levels[node]}",
            fontsize=8, color="blue"
        )
    nx.draw_networkx_edges(G, pos, edgelist=edges.keys(), width=2, edge_color="green", ax=ax, arrows=False)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

# Function to update the ant's position
def update_ant_position(frame):
    draw_graph()
    
    # Determine the current edge index and the progress along the edge
    current_edge = frame // frames_per_edge
    t = (frame % frames_per_edge) / frames_per_edge  # Fraction along the edge
    
    # Ensure we don't exceed the path length
    if current_edge >= len(path) - 1:
        return  # Stop moving if the ant has reached the final node
    
    # Calculate positions for the current edge
    start_node = path[current_edge]
    end_node = path[current_edge + 1]
    start_x, start_y = pos[start_node]
    end_x, end_y = pos[end_node]
    
    # Interpolate the ant's position
    ant_x = (1 - t) * start_x + t * end_x
    ant_y = (1 - t) * start_y + t * end_y
    
    # Draw the ant at the interpolated position
    ax.plot(ant_x, ant_y, 'ro', markersize=10)

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
draw_graph()

# Create animation
ani = animation.FuncAnimation(fig, update_ant_position, frames=total_frames, interval=200)
ani.save("aco_send_data.gif", writer="pillow", fps=10)

plt.show()


