import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import math

# Node Class
class Node:
    def __init__(self, node_id, x, y, energy, initial_pheromone, n, max_num_ants):
        self.node_id = node_id                      # ID of the node
        self.location = (x,y)                       # (x,y) location of the node
        self.energy = energy                        # Energy of the node
        self.neighbors = []                         # List of neighbors (node ids)
        self.num_neighbors = 0                      # Number of neighbors
        self.pheromone_matrix = np.full((n, n), initial_pheromone)    # Pheromone matrix (n x n)
        self.loss_matrix = np.zeros((n, n))         # Loss percentage matrix (n x n)
        self.max_num_ants = max_num_ants            # Max number of ants to be produced
        self.generated_ants_count = 0               # Number of ants generated
        self.ants = []                              # List of ants currently at this node

    def add_neighbor(self, neighbor_id,  loss_percent=0):
        self.neighbors.append(neighbor_id)
        self.loss_matrix[self.node_id][neighbor_id] = loss_percent
        self.num_neighbors = len(self.neighbors)

    def clear_k_ants(self):
        """
        Initializes the node by resetting the list of ants for the next iteration.
        Clears all previously stored ants.
        """
        self.ants = []  # Clear the ants list
        self.generated_ants_count = 0  # Reset the generated ants count
        #print(f"Node {self.node_id}: Initialized for new ants.")

    def generate_ant(self, source_node_id, sink_node_id, pheromone_matrix):
        """
        Generates one ant if the node has not yet reached its max_num_ants.
        Updates the internal list of ants and increments the generated_ants_count.

        :param source_node: The ID of the source node where the ant is generated.
        :param sink_node: The ID of the sink node (destination).
        :param pheromone_matrix: The pheromone matrix that will be copied to the ant.
        """
        # Only generate an ant if the count is less than max_num_ants
        if self.generated_ants_count < self.max_num_ants:
            # Create a new ant instance
            ant_id = self.generated_ants_count + 1  # Assign an ID to the ant
            new_ant = Ant(ant_id, source_node_id, sink_node_id, pheromone_matrix)
            
            # Add the ant to the list of ants at this node
            self.ants.append(new_ant)
            
            # Increment the count of generated ants
            self.generated_ants_count += 1
            
            #print(f"Ant {new_ant.ant_id} generated at Node {self.node_id}.")
        else:
            print(f"Node {self.node_id} has reached the max number of ants.")



# Ant Class
class Ant:
    def __init__(self, ant_id, source_node_id, sink_node_id, pheromone_matrix):
        self.ant_id = ant_id                        # Unique ID of the ant
        self.source_node_id = source_node_id              # Source node of the ant
        self.sink_node_id = sink_node_id                  # Destination node (sink)
        self.visited_nodes = [source_node_id]          # List of nodes visited, starting from the source
        self.current_position = source_node_id         # Current position (initially at source)
        self.distance_traveled = 0.0                # Total distance traveled
        self.curr_back_index=-1                     # Used in back ant how much path retraced
        self.is_back_ant = False                    # Indicates if the ant is retracing its path (True if retracing)
        self.dropped = False
        self.pheromone_matrix = [row[:] for row in pheromone_matrix]  # Copy of the source node's pheromone matrix


    # Function to move the ant to the next hop
    def move_ant(self, nodes, alpha=1.0, beta=2.0, m=0.3):
        if self.dropped:
            #ant reached a dead end
            return
        
        if self.is_back_ant:
            # Moving backward by using the index and updating the distance list
            if self.curr_back_index > 0:  # Ensure there's still a path to backtrack            
                # Update distance traveled by the distance between current and previous node
                prev_node_id = self.visited_nodes[self.curr_back_index - 1]
                curr_node_id = self.visited_nodes[self.curr_back_index]
                self.distance_traveled += euclidean_distance(nodes[curr_node_id], nodes[prev_node_id])
                next_hop=prev_node_id
                # Move to the previous node in the visited path
                self.curr_back_index -= 1
                self.current_position = prev_node_id
                
                #print(f"Back Ant {self.ant_id} moved to node {self.current_position}, total distance: {self.distance_traveled}")
            
            else:
                # If current back index is 0, it means we've reached the source node
                #print(f"Ant {self.ant_id} has reached the source node while retracing.")
                return  # Ant has finished its journey back to the source

        else:
            # Calculate next hop using probabilities during forward movement
            current_node = nodes[self.current_position]
            probabilities = get_next_hop_probabilities(current_node, self, nodes, alpha, beta, m)
            
            if(len(probabilities)==0 or sum([prob for _, prob in probabilities])==0):
                self.dropped=True
                return

            next_hop = choose_next_hop(probabilities)
            self.visited_nodes.append(next_hop)

            if next_hop == self.sink_node_id:
                self.start_backtracking()
                #print(f"Ant {self.ant_id} reached the Sink Node {self.sink_node_id}. Backtracking started.")

        current_node = nodes[self.current_position]
        self.distance_traveled += euclidean_distance(current_node, nodes[next_hop])
        self.current_position = next_hop

    def start_backtracking(self):
        self.is_back_ant = True
        self.curr_back_index=len(self.visited_nodes)-1
        #print(f"Ant {self.ant_id} is now retracing its path back to the source.")


# def store_ant(ant, source_node):
#     """
#     Stores the ant in the source node after it completes its journey back.
#     """
#     source_node.ants.append(ant)
#     #print(f"Ant {ant.ant_id} has been stored at the source node {source_node.node_id}.")


# Function to evaporate pheromones for all neighbors of a node, excluding visited nodes
def evaporate_pheromones(source_node, rho, tau_min):
    """
    Evaporates pheromone levels for all the pheromone matrix entries stored in the source node.
    The evaporation applies to all entries by multiplying them with (1 - rho) and ensures 
    pheromone values do not fall below tau_min.

    :param source_node: The node that stores the pheromone matrix (source node).
    :param rho: Evaporation rate.
    :param tau_min: Minimum pheromone level.
    """
    n = len(source_node.pheromone_matrix)  # Assuming pheromone matrix is n x n in size

    # Iterate over the entire pheromone matrix
    for i in range(n):
        for j in range(n):
            # Evaporate pheromone value
            tau_ij = source_node.pheromone_matrix[i][j]
            new_tau_ij = (1 - rho) * tau_ij  # Apply evaporation

            # Ensure the pheromone doesn't fall below tau_min
            if new_tau_ij < tau_min:
                new_tau_ij = tau_min

            # Update the pheromone value in the matrix
            source_node.pheromone_matrix[i][j] = new_tau_ij

            # #print updated pheromone information for debugging
            #print(f"Pheromone evaporated between Node {i} and Node {j}: {tau_ij} -> {new_tau_ij}")

def add_pheromone_from_ants(source_node, sink_node, tau_max):
    """
    Adds pheromones to the source node's pheromone matrix based on the paths taken by the ants.
    For each ant, the function iterates through its visited nodes and adds Q / Lk to the pheromone
    levels between consecutive nodes in the path, ensuring the pheromone level does not exceed tau_max.

    :param source_node: The node that stores the pheromone matrix (source node).
    :param Q: Constant value used to calculate the pheromone contribution (default is 1.0).
    :param tau_max: Maximum allowed pheromone level (upper bound).
    """
    Q = 0
    ants = source_node.ants
    
    if(len(ants)):
        Q = 1.5* euclidean_distance(source_node, sink_node)

    for ant in ants:
        # Ensure the ant has a valid path and has completed its journey
        if len(ant.visited_nodes) < 2 or ant.distance_traveled <= 0 or ant.dropped:
            continue  # Skip ants that don't have a valid path or didn't travel

        # Calculate the pheromone contribution for this ant
        pheromone_contribution = Q / ant.distance_traveled

        # Update pheromones for each edge in the visited path
        for i in range(len(ant.visited_nodes) - 1):
            node_i = ant.visited_nodes[i]
            node_j = ant.visited_nodes[i + 1]

            # Add the pheromone contribution to both directions (i->j and j->i)
            source_node.pheromone_matrix[node_i][node_j] += pheromone_contribution
            source_node.pheromone_matrix[node_j][node_i] += pheromone_contribution

            # Apply the upper bound (tau_max) to ensure pheromone levels don't exceed tau_max
            if source_node.pheromone_matrix[node_i][node_j] > tau_max:
                source_node.pheromone_matrix[node_i][node_j] = tau_max
            if source_node.pheromone_matrix[node_j][node_i] > tau_max:
                source_node.pheromone_matrix[node_j][node_i] = tau_max

 
# Function to calculate Euclidean distance between two nodes
def euclidean_distance(node1, node2):
    return math.sqrt((node1.location[0] - node2.location[0]) ** 2 + (node1.location[1] - node2.location[1]) ** 2)


# Function to generate n nodes randomly in a 10x10 area and set neighbors based on distance
def initialize_nodes(n, side_length, r_min, r_max, initial_energy, max_num_ants, initial_pheromone):
    """
    Initializes the nodes with unique random locations within a square of side_length x side_length,
    ensuring no two nodes are at the same position, no neighbors are within r_min radius, 
    and each node has at least one neighbor within r_max radius.
    """
    nodes = []
    positions = set()

    # Adding the source node at position (0, 0)
    source_node = Node(node_id=0, x=0, y=0, energy=initial_energy, initial_pheromone=initial_pheromone, n=n, max_num_ants=max_num_ants)  # Infinite energy for source
    positions.add((0, 0))
    nodes.append(source_node)

    # Adding the sink node at position (side_length, side_length)
    sink_node = Node(node_id=1, x=side_length, y=side_length, energy=initial_energy,  initial_pheromone=initial_pheromone, n=n, max_num_ants=max_num_ants)  # Infinite energy for sink
    positions.add((side_length, side_length))
    nodes.append(sink_node)

    # Randomly position the remaining nodes, avoiding r_min distance from existing nodes
    while len(nodes) < n:
        x, y = round(random.uniform(0, side_length), 2), round(random.uniform(0, side_length), 2)
        if (x, y) not in positions:
            valid_position = True
            for node in nodes:
                distance = math.sqrt((node.location[0] - x) ** 2 + (node.location[1] - y) ** 2)
                if distance < r_min:  # Ensure new node is not within r_min of existing nodes
                    valid_position = False
                    break
            if valid_position:
                positions.add((x, y))
                new_node = Node(node_id=len(nodes), x=x, y=y, energy=initial_energy,  initial_pheromone=initial_pheromone, n=n, max_num_ants=max_num_ants)
                nodes.append(new_node)

    # Now calculate neighbors based on communication radius r_max and ensure no neighbors within r_min
    for node in nodes:
        neighbors = []
        for other_node in nodes:
            if node.node_id != other_node.node_id:
                distance = euclidean_distance(node, other_node)
                if r_min <= distance <= r_max:
                    neighbors.append(other_node.node_id)
        # Assign neighbors to the node if there's at least one within r_max
        if neighbors:
            for neighbor_id in neighbors:
                node.add_neighbor(neighbor_id)

    return nodes


# Function to initialize ants with a given source node and sink node
def initialize_ants(source_node_id, nodes, sink_node_id):
    nodes[source_node_id].generate_ant(source_node_id,sink_node_id,nodes[source_node_id].pheromone_matrix)


# Function to calculate the eta value for a given neighbor
def get_eta(current_node, neighbor_node, source_node, sink_node, m):
    """
    Calculate the eta value for a neighbor.
    
    :param current_node: The node where the ant is currently located.
    :param neighbor_node: The neighbor node being evaluated.
    :param sink_node: The sink node (destination of the ant).
    :param m: Weight factor for distance calculation.
    
    :return: The eta value based on distance, energy, and number of neighbors.
    """
    dist_source_sink = euclidean_distance(source_node, sink_node)
    dist_i_j = euclidean_distance(current_node, neighbor_node)
    dist_j_sink = euclidean_distance(neighbor_node, sink_node)
    
    energy_ratio=0

    if(current_node.energy!=0):
        energy_ratio = neighbor_node.energy / current_node.energy
    max_neighbors = max([node.num_neighbors for node in nodes])  # Max neighbors across all nodes
    
    eta = 5*(dist_source_sink / (m * dist_i_j + (1 - m) * dist_j_sink)) * (energy_ratio) * ((neighbor_node.num_neighbors / max_neighbors)**0.5)
    #print(eta, "hehee")
    return eta


# Function to calculate the probability list for choosing the next hop
def get_next_hop_probabilities(node, ant, nodes, alpha, beta, m):
    """
    This function calculates the probability list for choosing the next hop for an ant.
    It returns both the node IDs of the neighbors and the corresponding probabilities.
    
    :param node: The current node where the ant is located.
    :param ant: The ant instance.
    :param nodes: List of all nodes in the network.
    :param alpha: The importance of pheromone (pheromone influence).
    :param beta: The importance of eta (visibility).
    :param m: Weight factor for eta calculation.
    :return: A list of tuples (neighbor_id, probability) for each neighbor.
    """
    pheromones = []
    etas = []
    probabilities = []
    current_id = node.node_id
    source_node = nodes[ant.source_node_id]  # Source node
    sink_node = nodes[ant.sink_node_id]  # Sink node
    valid_neighbors = []  # List of neighbors that the ant can visit (not already visited)

    # Collect the pheromones and eta values for neighbors
    for neighbor_id in node.neighbors:
        if neighbor_id not in ant.visited_nodes:  # Avoid returning to visited nodes
            neighbor_node = nodes[neighbor_id]
            pheromone_level = ant.pheromone_matrix[current_id][neighbor_id]
            eta_value = get_eta(node, neighbor_node, source_node, sink_node, m)
            pheromones.append(pheromone_level ** alpha)  # Pheromone influence
            etas.append(eta_value ** beta)               # Eta influence
            valid_neighbors.append(neighbor_id)          # Add valid neighbor ID
        # else:
        #     pheromones.append(0)  # No pheromone for visited nodes
        #     etas.append(0)        # No eta for visited nodes

    # Compute the product of pheromone influence and eta for each neighbor
    combined_influences = [pheromones[i] * etas[i] for i in range(len(pheromones))]

    total_influence = sum(combined_influences)
    #print(total_influence)
    #print("aryan")
    # If the total influence is zero, return zero probabilities
    if total_influence == 0:
        return [(neighbor_id, 0) for neighbor_id in valid_neighbors]
    
    # Calculate the probability for each neighbor
    probabilities = [(valid_neighbors[i], combined_influences[i] / total_influence) for i in range(len(valid_neighbors))]
    #print(probabilities)
    #print(etas)
    #print(pheromones)
    #print("maurya")
    return probabilities


# Function to select the next hop based on the probability distribution and return the next node ID
def choose_next_hop(probability_list):
    """
    Function to select the next hop based on the provided probability distribution.
    
    :param probability_list: A list of tuples (neighbor_id, probability) where each entry represents 
                             a neighbor node and the probability of selecting that neighbor.
    
    :return: The node ID of the chosen neighbor.
    """
    neighbors = [neighbor_id for neighbor_id, _ in probability_list]
    probabilities = [prob for _, prob in probability_list]
    
    # Use random.choices to select a neighbor based on the given probabilities
    chosen_index = random.choices(range(len(neighbors)), weights=probabilities, k=1)[0]
    
    return neighbors[chosen_index]


def establish_route(source_node_id, sink_node_id,  nodes, rho, tau_min, tau_max):
    """
    Establish a route by generating k ants from source to sink and back.
    Move ants and generate new ants after every few steps until all k ants are generated.
    
    :param source_node_id: ID of the source node
    :param sink_node_id: ID of the sink node
    :param k_ants: Number of ants to generate
    :param nodes: List of all nodes in the network
    :param moves_per_generation: Number of moves each ant takes before new ants are generated
    """
    # initialize_ants(source_node_id, nodes, sink_node_id)

    while nodes[source_node_id].generated_ants_count < nodes[source_node_id].max_num_ants:
        # Generate ants if we haven't reached the target k ants
        initialize_ants(source_node_id, nodes, sink_node_id)
        #print(f"Generated Ant at Node {source_node_id}")

    # Move each ant 
    for ant in nodes[source_node_id].ants:
        while ant.curr_back_index!=0 and not ant.dropped:
            #print("aryan")
            #print(ant.curr_back_index)
            #print(ant.visited_nodes)
            ant.move_ant(nodes)
    
    #When all ants reach the source node then update pheromone and remove ants
    evaporate_pheromones(nodes[source_node_id], rho, tau_min)
    add_pheromone_from_ants(nodes[source_node_id], nodes[sink_node_id], tau_max)

    # Initialize ants for the next round
    nodes[source_node_id].clear_k_ants()

def generate_data_path(source_node, sink_node_id, nodes):
    """
    Generates the path from the source node to the sink node based on the highest pheromone levels.
    
    :param source_node: The starting node (source) for the path.
    :param sink_node_id: The ID of the sink node (destination).
    :param nodes: The list of all nodes in the network.
    
    :return: A list of node IDs representing the path from source to sink based on the highest pheromone levels.
    """
    current_node_id = source_node.node_id  # Start from the source node
    path = [current_node_id]  # Initialize the path with the source node

    while current_node_id != sink_node_id:
        current_node = nodes[current_node_id]
        
        # Find the neighbor with the highest pheromone value
        max_pheromone = -1
        next_node_id = None
        
        for neighbor_id in current_node.neighbors:
            pheromone_level = source_node.pheromone_matrix[current_node_id][neighbor_id]
            
            if pheromone_level > max_pheromone and neighbor_id not in path:
                max_pheromone = pheromone_level
                next_node_id = neighbor_id

        if next_node_id is None:
            #print("Error: No path found with pheromone levels leading to the sink node.")
            return path  # If no next node is found, return the path constructed so far
        
        # Add the next node with the highest pheromone to the path
        path.append(next_node_id)
        current_node_id = next_node_id
        
        # Debug #print statement
        #print(f"Moving to Node {next_node_id} with pheromone level {max_pheromone}")
    
    # Return the final path when we reach the sink node
    return path

def send_data(source_node_id, sink_node_id, nodes, data, Elec, epsilon, rho, tau_min, tau_max):
    """
    Sends a data packet from source to sink. Uses pheromone levels to decide the next hop
    while considering the packet loss ratio from the loss matrix of the current node.
    
    :param source_node_id: ID of the source node
    :param sink_node_id: ID of the sink node
    :param nodes: List of all nodes in the network
    :param data: The data to be sent in the packet
    :param Elec: Transmission energy
    :param epsilon: Amplification energy
    """
    sent_packets = []
    received_packets = []
    
    # Initialize the data packet with source, sink, current position, path, and the data field.
    optimal_path=generate_data_path(source_node, sink_node_id, nodes)
    optimal_path.reverse()
    packet = {
        'source': source_node_id,
        'sink': sink_node_id,
        'current_position': source_node_id,
        'path': optimal_path,
        'data': data,
        'lost': False,
        'size': 800
    }
    packet['path'].pop()
    sent_packets.append(packet)
    #print(f"Data packet sent from Source Node {source_node_id} to Sink Node {sink_node_id} with data: {data}")

    # Move the packet through the network
    while packet['current_position'] != sink_node_id:
        next_hop = move_data(packet, nodes)
        if next_hop is None:  # Packet dropped due to loss
            packet['lost'] = True
            #print(f"Packet dropped while moving from Node {packet['current_position']}.")
            return sent_packets, received_packets

        #Update Energy values
        # print("Curr: ", packet['current_position'], " next: ", next_hop)
        update_energy(Elec, epsilon, packet['size'], nodes[packet['current_position']], nodes[next_hop], nodes, rho, tau_min, tau_max)

        # Update the packet's path and current position
        packet['path'].pop()
        packet['current_position'] = next_hop

    # If it reaches the sink, add it to the received packets list
    #print(f"Packet successfully reached Sink Node {sink_node_id} with data: {packet['data']}")
    received_packets.append(packet)

    return sent_packets, received_packets


def move_data(packet, nodes):
    """
    Moves the data packet to the next hop based on the highest pheromone neighbor.
    Takes packet loss ratio into account from the current node's loss matrix before making the decision.
    
    :param packet: The current data packet being moved
    :param nodes: List of all nodes in the network
    :return: The ID of the next hop node or None if packet is dropped due to loss
    """
    current_node = nodes[packet['current_position']]
    next_hop = None
    
    loss_ratio = current_node.loss_matrix[current_node.node_id][packet['path'][-1]]

    # Packet can only be sent if the random value exceeds the loss ratio
    if random.random() > loss_ratio:
        next_hop = packet['path'][-1]

    if next_hop is not None:
        #print(f"Moving packet to Node {next_hop} from Node {packet['current_position']}.")
        return next_hop  # Return the next node to which the packet should move
    else:
        #print(f"Packet lost due to high loss ratio from Node {packet['current_position']}.")
        return None  # Packet dropped due to loss

def update_energy(Eelec, epsilon, l, transmiting_node, receiving_node, nodes, rho, tau_min, tau_max):
    """
    Calculate the total energy consumption for wireless communication.
    
    Parameters:
    Eelec (float): Energy consumption per bit (in Joules per bit).
    epsilon (float): Path loss factor (in Joules per bit per meter^4).
    l (int): Number of bits of data being transmitted.
    
    Returns:
    float: Total energy consumption in Joules.
    """

    d = euclidean_distance(transmiting_node,receiving_node)/500

    # Energy due to transmission of l bits
    energy_transmission = Eelec * l
    
    # Energy due to multipath fading (d^4 term)
    energy_fading = epsilon * l * (d ** 4)
    
    # Total energy
    total_energy = energy_transmission + energy_fading
    # print("In Energy Curr: ", transmiting_node.node_id, " next: ", receiving_node.node_id)

    if receiving_node.node_id > 1 :
        receiving_node.energy-=energy_transmission
    if transmiting_node.node_id > 1:
        transmiting_node.energy-=total_energy

    if(receiving_node.energy<=0):
        remove_node(nodes, receiving_node.node_id, rho, tau_min, tau_max)
        

    if(transmiting_node.energy<=0):
        remove_node(nodes, transmiting_node.node_id, rho, tau_min, tau_max)

    
    return 

def plot_graph(nodes, path1, path2=[], show_neighbors=True):
    plt.figure(figsize=(6, 6))
    
    # Plot the edges
    if show_neighbors:
        for node in nodes:
            x, y = node.location  # Get the (x, y) coordinates of the current node
            for neighbor_id in node.neighbors:
                neighbor = nodes[neighbor_id]  # Get the neighbor node
                x_n, y_n = neighbor.location   # Neighbor's coordinates
                # Plot edge between the node and its neighbor
                plt.plot([x, x_n], [y, y_n], color='black', linestyle='-', linewidth=1)

    # Plot the nodes
    for i, node in enumerate(nodes):
        x, y = node.location
        if i == 0:  # Source node
            plt.scatter(x, y, color='blue', s=100, label='Source' if i == 0 else "")  # Blue for source
        elif i == 1:  # Sink node
            plt.scatter(x, y, color='green', s=100, label='Sink' if i == 1 else "")  # Green for sink
        else:
            plt.scatter(x, y, color='red', s=100)  # Red for all other nodes
        # Add labels to the nodes (node_id)
        plt.text(x , y, f'{node.node_id}', fontsize=12)

    # Plot the path in green
    for k in range(len(path1) - 1):
        node1 = nodes[path1[k]]     # Current node
        node2 = nodes[path1[k + 1]] # Next node in the path
        x1, y1 = node1.location
        x2, y2 = node2.location
        # Plot a green edge between adjacent nodes in the path
        plt.plot([x1, x2], [y1, y2], color='green', linestyle='-', linewidth=2)

     # Plot the path in green
    for k in range(len(path2) - 1):
        node1 = nodes[path2[k]]     # Current node
        node2 = nodes[path2[k + 1]] # Next node in the path
        x1, y1 = node1.location
        x2, y2 = node2.location
        # Plot a green edge between adjacent nodes in the path
        plt.plot([x1, x2], [y1, y2], color='red', linestyle='-', linewidth=2)

    # Set plot labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Graph of Nodes and Edges")

    # Show the legend for source and sink
    plt.legend()

    # Display grid
    plt.grid(True)

    # Show the plot for 0.5 seconds and then close
    plt.show()
    # plt.pause(0.5)
    # plt.close()


def remove_node(nodes, x, rho, tau_min, tau_max):
    nodes[x].energy=0
    for neighbor in nodes[x].neighbors:
        nodes[0].pheromone_matrix[neighbor][x]=0
        nodes[0].pheromone_matrix[x][neighbor]=0
    for i in range(5):
        establish_route(0, 1, nodes, rho, tau_min, tau_max)
    
    print("New path: ", generate_data_path(nodes[0],1,nodes))

def priint(matrix):
    print("Pheromone Matrix:")
    for row in matrix:
        print([f"{value:.2f}" for value in row])

def priint_energies(nodes):
    for node in nodes:
        print(node.node_id, node.energy)
        
# Example of setting up the network, initializing ants, and moving ants
n = 20              # Number of nodes
side_length = 500    # Side length of square area of network
energy = 120        # Same energy level for all nodes
r_min = 100            # Neighbor node range
r_max = 250            # Neighbor node range
num_ants = 50        # Number of ants
initial_pheromone=10.0
rho = 0.1
tau_max = 40
tau_min = 2
Elec = 10**(-5)
epsilon = 10**(-6)


# Generate nodes
nodes = initialize_nodes(n,side_length, r_min, r_max,energy, num_ants, initial_pheromone)
source_node=nodes[0]
sink_node=nodes[1]
path=[]

plot_graph(nodes, path)

for i in range(5):
    establish_route(source_node.node_id, sink_node.node_id, nodes, rho, tau_min, tau_max)
    # priint(source_node.pheromone_matrix)
    path=generate_data_path(source_node, sink_node.node_id,nodes)
    #print(path)
    plot_graph(nodes, path, show_neighbors=False)
print("Original Path: ", path)
for i in range (19000):
    send_data(0,1,nodes,"",Elec,epsilon, rho, tau_min, tau_max)

priint_energies(nodes)
path1=path
for i in range(5):
    establish_route(source_node.node_id, sink_node.node_id, nodes, rho, tau_min, tau_max)
    # priint(source_node.pheromone_matrix)
    path=generate_data_path(source_node, sink_node.node_id,nodes)
    #print(path)
    plot_graph(nodes, path, show_neighbors= False)

plot_graph(nodes, path1, path, False)

# priint_energies(nodes)
