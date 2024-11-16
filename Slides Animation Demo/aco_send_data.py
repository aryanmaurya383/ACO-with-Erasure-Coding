import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

# Define the graph structure with forward and reverse edges
G = nx.DiGraph()
edges = {
    (0, 1): 5, (0, 2): 8, (0, 3): 12,
    (1, 4): 6, (1, 5): 9,
    (2, 4): 6, (2, 5): 9,
    (3, 4): 6, (3, 5): 9,
    (4, 6): 6, (5, 6): 3,
    # Reverse edges to allow bidirectional travel
    (1, 0): 5, (2, 0): 8, (3, 0): 12,
    (4, 1): 6, (5, 1): 9,
    (4, 2): 6, (5, 2): 9,
    (4, 3): 6, (5, 3): 9,
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

# Initialize pheromone levels for edges (scaled to 0 initially)
pheromone_levels = {edge: 0.1 for edge in G.edges}

# Initialize energy levels for nodes (starts at 1 for each node)
node_energy = {node: 1 for node in G.nodes}

# Define the paths for ants (forward and reverse)
path1 = [(0, 1), (1, 4), (4, 6)]  # Path for the first ant (source to sink)
path2 = [(0, 2), (2, 5), (5, 6)]  # Path for the second ant (different path)
path_actual = [
    [(0, 1), (1, 4), (4, 6)],
    [(0, 1), (1, 4), (4, 6)],
    [(0, 2), (2, 5), (5, 6)],
    [(0, 2), (2, 5), (5, 6)]
]

# Additional paths for 6 ants (3 paths and their reverse)
paths_6_ants = [
    [(0, 1), (1, 4), (4, 6)],  # Path 1
    [(0, 2), (2, 5), (5, 6)],  # Path 2
    [(0, 3), (3, 4), (4, 6)],  # Path 3
    [(0, 2), (2, 4), (4, 6)],  # Path 4 (shorter path)
    [(0, 3), (3, 5), (5, 6)],  # Path 5
    [(0, 1), (1, 5), (5, 6)]   # Path 6 (shorter path)
]

# Pre-calculate the pheromone levels and node energy consumption after all ants have completed their paths
def pre_calculate():
    global pheromone_levels, node_energy
    
    # Simulate all the ants' movements
    for path in [path1, path2] + paths_6_ants:
        # Forward pass
        for edge in path:
            pheromone_levels[edge] += 0.7  # Increase pheromone on the edge
  
pre_calculate()
string_text=''
# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
X_VAL=1
Y_VAL=1.2
pheromone_text = ax.text(X_VAL, Y_VAL, string_text, ha='center', va='center', fontsize=15, color="rebeccapurple", zorder=1000)


# Function to draw the graph with updated pheromone levels and node energy
def draw_graph():
    global string_text, X_VAL, Y_VAL
    ax.clear()  # Clear the axis for a fresh plot
    node_colors = ['slategray' if node == 0 else 'red' if node == 6 else 'lightblue' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, ax=ax, arrows=False)    
    # Draw edges with thickness based on dynamic pheromone levels
    ax.text(X_VAL, Y_VAL, string_text, ha='center', va='center', fontsize=15, color="rebeccapurple", zorder=1000)

    for edge, weight in edges.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=[edge],
            width=pheromone_levels[edge] * 2,  # Thickness proportional to pheromone level
            edge_color="green",
            ax=ax,
            arrows=False  # No arrows to make edges appear undirected
        )
        
    # Draw edge labels for weights
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    
    # Draw node energy levels below the nodes
    node_labels = {node: f"{node_energy[node]:.2f}" for node in G.nodes()}
    
    # Offsetting energy labels slightly below the nodes
    energy_pos = {node: (x, y - 0.15) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, energy_pos, font_color="deeppink", labels=node_labels, font_size=10, ax=ax)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='slategray', markersize=10, label='Source'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Sink'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', markersize=10, label='Data Packet'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='deeppink', markersize=10, label='Node energy'),
            plt.Line2D([0], [0], color='green', lw=2, label='Pheromone Trail'),
        ],
        loc='best',
        fontsize=10
    )

# Initialize the moving ants (starting with one red dot)
ants = [{'ant': ax.plot(*pos[0], "o",color="coral", markersize=8)[0], 'path': path1, 'index': 0, 'reverse': False}]  # First ant at the source node

count = 0
flag=0

# Function to update frame in animation
def update(frame_num):
    global count, flag, string_text
    RATE = 10
    step = frame_num // (2*RATE)  # Current step in the path
    edge_progress = (frame_num % RATE) / RATE  # Progress along the current edge (0 to 1)

    if(count<2):
        # Move each ant along its path
        for ant_data in ants[:]:
            ant = ant_data['ant']
            path = ant_data['path']
            reverse = ant_data['reverse']
            
            # Handle the reverse direction when the ant reaches the sink node
            if ant_data['index'] == len(path) and not reverse:
                ant_data['reverse'] = True
                path = path[::-1]  # Reverse the path
                path = [(b, a) for a, b in path]
                ant_data['path'] = path
                ant_data['index'] = 0  # Reset to start of the reverse path

            # Determine which edge the ant is on and move it along that edge
            start, end = path[ant_data['index']]
            x_start, y_start = pos[start]
            x_end, y_end = pos[end]
            
            
            # Calculate ant position along the current edge
            ant.set_data(
                x_start + (x_end - x_start) * edge_progress,
                y_start + (y_end - y_start) * edge_progress
            )

            # Update the index for the next step along the path
            if edge_progress == 1 - 1 / RATE:  # Ant has reached the end of the current edge
                ant_data['index'] += 1
                #decrease energy by 01
                node_energy[start] -= 0.05
                node_energy[end] -= 0.05
            
            # Remove ants that have completed their journey (both forward and reverse)
            if ant_data['reverse'] and ant_data['index'] == len(path):
                ants.pop()  # Remove ant from the animation
                count+=1
                ants.append({'ant': ax.plot(*pos[0], "ro", markersize=8)[0], 'path': path_actual[count], 'index': 0, 'reverse': False})
    elif (flag<2*RATE):
        string_text="Pheromone Level Decreasing"
        value_=1/(2*RATE)
        pheromone_levels[(0, 1)] -= value_
        pheromone_levels[(1, 4)] -= value_
        pheromone_levels[(4,6)] -= value_
        flag+=1
    elif (count<4):
        string_text=''
        for ant_data in ants[:]:
            ant = ant_data['ant']
            path = ant_data['path']
            reverse = ant_data['reverse']
            
            # Handle the reverse direction when the ant reaches the sink node
            if ant_data['index'] == len(path) and not reverse:
                ant_data['reverse'] = True
                path = path[::-1]  # Reverse the path
                path = [(b, a) for a, b in path]
                ant_data['path'] = path
                ant_data['index'] = 0  # Reset to start of the reverse path

            # Determine which edge the ant is on and move it along that edge
            start, end = path[ant_data['index']]
            x_start, y_start = pos[start]
            x_end, y_end = pos[end]
            
            
            # Calculate ant position along the current edge
            ant.set_data(
                x_start + (x_end - x_start) * edge_progress,
                y_start + (y_end - y_start) * edge_progress
            )

            # Update the index for the next step along the path
            if edge_progress == 1 - 1 / RATE:  # Ant has reached the end of the current edge
                ant_data['index'] += 1
                #decrease energy by 01
                node_energy[start] -= 0.05
                node_energy[end] -= 0.05
            
            # Remove ants that have completed their journey (both forward and reverse)
            if ant_data['reverse'] and ant_data['index'] == len(path):
                ants.remove(ant_data)  # Remove ant from the animation
                count+=1
                if (count==3):
                    ants.append({'ant': ax.plot(*pos[0], "o",color="coral", markersize=8)[0], 'path': path_actual[count], 'index': 0, 'reverse': False})

    # Redraw the graph with updated pheromone levels and energy
    draw_graph()

    # Redraw the ants on top of the graph
    for ant_data in ants:
        ax.plot(*ant_data['ant'].get_data(), "o",color="coral", markersize=8)  # Redraw ants as red dots

# Create and run the animation
ani = animation.FuncAnimation(fig, update, frames=1000, interval=1)
ani.save("aco_send_data.gif", writer="pillow", fps=10)
plt.show()
