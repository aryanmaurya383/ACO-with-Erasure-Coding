import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import math
import copy
import matplotlib.animation as animation

all_nodes = []
all_paths = []

# Define constants for BOA parameters
sensory_modality = 0.01  # Sensory modality (c)
power_exponent = 0.5     # Power exponent (a)
switch_probability = 0.8  # Probability to choose between global and local search

# Fitness function weights (from earlier equations)
delta1, delta2, delta3, delta4, delta5 = 0.35, 0.25, 0.2, 0.1, 0.1

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
        self.cluster_locations = []                 # All neighboring sensor nodes locaiton for this CH 
        self.cluster_energies = []                  # All neighboring sensor nodes energies for this CH
        self.cluster_d_avg = []                     # All neighboring sensor nodes average distance with other nodes
        self.curr_CH = 0                            # index in the cluster_list of the current cluster head
        self.fragnance = 0                          # Fragnance of the node
        self.fragnances = []                        # Fragnance of the neighbors
        self.fitness = 0                            # Fitness of the node
        self.virtual_position = (x,y)               # Virtual position of the node

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
    
    def generate_random_node_within_radius(self, r_min, k):
        """Generates a random node within r_min radius of the given center node."""
        self.cluster_locations = [self.location]
        self.cluster_energies = [self.energy]
        for i in range(k):
            angle = 2 * math.pi * (i/k)  # Angle in radians
            radius = random.uniform(0.1* r_min, r_min)       # Random radius within r_min
            # Calculate new (x, y) coordinates based on the center node's location
            x_new = self.location[0] + radius * math.cos(angle)
            y_new = self.location[1] + radius * math.sin(angle)
            
            self.cluster_locations.append((x_new, y_new))
            self.cluster_energies.append(self.energy)

        #Update d_avg list
        for i, loc1 in enumerate(self.cluster_locations):
            total_distance = 0
            count = 0
            
            for j, loc2 in enumerate(self.cluster_locations):
                if i != j:  # Exclude the distance to itself
                    distance = math.sqrt((loc2[0] - loc1[0]) ** 2 + (loc2[1] - loc1[1]) ** 2)
                    total_distance += distance
                    count += 1
            
            # Calculate average distance for the current location
            avg_distance = total_distance / count if count > 0 else 0
            self.cluster_d_avg.append(avg_distance)

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
def initialize_nodes(n, side_length, r_min, r_max, initial_energy, max_num_ants, initial_pheromone, cluster_size=5):
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
        #Genrate random cluster 
        node.generate_random_node_within_radius(r_min, cluster_size)
        
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

def send_data(source_node_id, sink_node_id, nodes, data,size, Elec, epsilon, rho, tau_min, tau_max):
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
    if(data==None):
        data=1


    # Initialize the data packet with source, sink, current position, path, and the data field.
    optimal_path=generate_data_path(nodes[source_node_id], sink_node_id, nodes)
    temp_path =copy.deepcopy(optimal_path)
    # if(optimal_path[-1]!=sink_node_id):
    #     return None, temp_path
    # # print(temp_path)
    optimal_path.reverse()
    packet = {
        'source': source_node_id,
        'sink': sink_node_id,
        'current_position': source_node_id,
        'path': optimal_path,
        'times': data,
        'lost': False,
        'size': size
    }
    # packet['path'].pop()
    dropped=0
    #print(f"Data packet sent from Source Node {source_node_id} to Sink Node {sink_node_id} with data: {data}")

    # Move the packet through the network
    while data>0:
        data-=1
        packet['source']=source_node_id
        packet['sink']=sink_node_id
        packet['current_position']=source_node_id
        packet['size']=400
        packet['lost']=False
        packet['path']=copy.deepcopy(optimal_path)
        packet['path'].pop()

        while packet['current_position'] != sink_node_id:
            if len(packet['path']) == 0:
                # nodes[packet['current_position']].energy=0
                break
            next_hop = move_data(packet, nodes)

            if next_hop is None:  # Packet dropped due to loss
                packet['lost'] = True
                break
                #print(f"Packet dropped while moving from Node {packet['current_position']}.")
                # return sent_packets, temp_path

            #Update Energy values
            update_energy(Elec, epsilon, packet['size'], nodes[packet['current_position']], nodes[next_hop], nodes, rho, tau_min, tau_max)

            # Update the packet's path and current position
            packet['path'].pop()
            packet['current_position'] = next_hop
        
        if(packet['lost']):
            dropped+=1

    # If it reaches the sink, add it to the received packets list
    #print(f"Packet successfully reached Sink Node {sink_node_id} with data: {packet['data']}")
    # received_packets.append(packet)
    check_energy_levels(nodes)

    return dropped, temp_path,

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

    d = euclidean_distance(transmiting_node,receiving_node)

    # Energy due to transmission of l bits
    energy_transmission = Eelec * l
    
    # Energy due to multipath fading (d^4 term)
    energy_fading = epsilon * l * (d ** 4)
    
    # Total energy
    total_energy = energy_transmission + energy_fading
    # print("In Energy Curr: ", transmiting_node.node_id, " next: ", receiving_node.node_id)

    if receiving_node.node_id != 1 :
        receiving_node.energy-=energy_transmission
    # if transmiting_node.node_id > 1:
    transmiting_node.energy-=total_energy

    if(receiving_node.energy<=0):
        receiving_node.energy=0
        select_new_CH_by_BAO(receiving_node, nodes[1].location)
        # remove_node(nodes, receiving_node.node_id, rho, tau_min, tau_max)
        

    if(transmiting_node.energy<=0):
        transmiting_node.energy=0
        select_new_CH_by_BAO(transmiting_node, nodes[1].location)
        # remove_node(nodes, transmiting_node.node_id, rho, tau_min, tau_max)

    
    return 

def check_energy_levels(nodes):
    """
    This function checks the energy levels of the current CH of each node in the cluster.
    If the CH's energy is less than half of the average energy of the cluster, it selects a new CH.
    """
    for node in nodes:
        # Calculate average energy of the cluster
        avg_cluster_energy = sum(node.cluster_energies) / len(node.cluster_energies)
        
        # Get the current CH's energy
        curr_ch_energy = node.energy
        
        # Check if the current CH's energy is less than half of the average cluster energy
        if curr_ch_energy < (0.5 * avg_cluster_energy):
            # Select a new CH
            select_new_CH_by_BAO(node, nodes[1].location)
            # path=generate_data_path(source_node, sink_node.node_id,nodes)
            # #print(path)
            # all_paths.append(path)  # Store the current path
            all_nodes.append(copy.deepcopy(nodes)) 
        if avg_cluster_energy == 0:
            remove_fully_dead_cluster(node, nodes)

def find_distance(location1, location2):
    return math.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


def remove_fully_dead_cluster(dead_node, nodes):
    dead_node.energy=0
    for node in nodes:
        for i in range(len(nodes)):
            node.pheromone_matrix[i][dead_node.node_id]=0
            node.pheromone_matrix[dead_node.node_id][i]=0


def select_new_CH_by_BAO(node, sink_node_location, max_iterations=100):
    new_ch_index = -1

    # Initialize fragrance dictionary to store neighbors' fragrances
    node.fragrances = {neighbor: 0 for neighbor in node.neighbors}
    node.cluster_energies[node.curr_CH] = node.energy

    # Calculate initial fitness and fragrance for the node
    f1 = 0 if node.energy == 0 else 1 / node.energy  # Residual energy
    f2 = sum(find_distance(node.location, ch) for ch in node.cluster_locations) / len(node.cluster_locations)
    f3 = find_distance(node.location, sink_node_location)
    f4 = len(node.neighbors)
    f5 = np.sqrt(sum(find_distance(node.location, n) ** 2 for n in node.cluster_locations) / len(node.neighbors))

    # Combined fitness (f) as per Eq. (12)
    node.fitness = delta1 * f1 + delta2 * f2 + delta3 * f3 + delta4 * f4 + delta5 * f5
    node.fragrance = sensory_modality * (node.fitness ** power_exponent)

    # Calculate and store fragrance for each neighbor
    for neighbor in node.neighbors:
        neighbor_fitness = delta1 * f1 + delta2 * f2 + delta3 * f3 + delta4 * f4 + delta5 * f5
        node.fragrances[neighbor] = sensory_modality * (neighbor_fitness ** power_exponent)

    # Main BOA loop for updating virtual positions
    for t in range(max_iterations):
        # Determine the best neighbor in the neighborhood (node and neighbors)
        best_neighbor = max([node] + node.neighbors, key=lambda n: node.fragrances.get(n, node.fragrance))
        g_star = best_neighbor.location

        r = random.random()
        if r < switch_probability:  # Global search phase
            node.virtual_position = node.location + r * (find_distance(g_star, node.location)) * node.fragrance
        else:  # Local search phase
            if len(node.neighbors) >= 2:
                j, k = random.sample(node.cluster_locations, 2)
                node.virtual_position = node.location + r * (find_distance(j, k)) * node.fragrance

        # Recalculate fitness and fragrance after position update
        f1 = 0 if node.energy == 0 else 1 / node.energy
        f2 = sum(find_distance(node.virtual_position, ch) for ch in node.cluster_locations) / len(node.cluster_locations)
        f3 = find_distance(node.virtual_position, sink_node_location)
        f4 = len(node.neighbors)
        f5 = np.sqrt(sum(find_distance(node.virtual_position, n) ** 2 for n in node.cluster_locations) / len(node.neighbors))

        # Skip updating fragrance if energy is zero
        if node.energy == 0:
            continue
        
        node.fitness = delta1 * f1 + delta2 * f2 + delta3 * f3 + delta4 * f4 + delta5 * f5
        node.fragrance = sensory_modality * (node.fitness ** power_exponent)
        
        for neighbor in node.neighbors:
            neighbor_fitness = delta1 * f1 + delta2 * f2 + delta3 * f3 + delta4 * f4 + delta5 * f5
            node.fragrances[neighbor] = sensory_modality * (neighbor_fitness ** power_exponent)
        
        # Select the best node based on fragrance and update CH if fragrance is higher
        for i, neighbor in enumerate(node.neighbors):
            if node.fragrances[neighbor] > node.fragrance:
                node.fragrance = node.fragrances[neighbor]
                node.location = node.cluster_locations[i]
                node.energy = node.cluster_energies[i]
                node.curr_CH = i

    return node

def select_new_CH(node, sink_node_location):
    """
    This function selects a new CH for the node based on the index I.
    The selection is based on residual energy, distance to sink, and average distance to other nodes.
    """
    max_index = 0
    new_ch_index = -1
    lambda_val = 1  # Assuming lambda is a constant, set it to 1 for now
    
    node.cluster_energies[node.curr_CH]=node.energy

    for i in range(len(node.cluster_locations)):
        E_res = node.cluster_energies[i]  # Residual energy of node i
        dis = find_distance(sink_node_location, node.cluster_locations[i])  # Distance to sink node
        d_avg = node.cluster_d_avg[i]  # Average distance of node i to other nodes
        
        # Calculate the index I for node i
        I_i = (lambda_val * E_res) / (dis * d_avg)
        
        # Select the node with the maximum I value
        if I_i > max_index:
            max_index = I_i
            new_ch_index = i
    
    # Update the CH with the new CH's location and energy, keeping other things the same
    if new_ch_index != -1:
        new_ch_location = node.cluster_locations[new_ch_index]
        new_ch_energy = node.cluster_energies[new_ch_index]
        
        # Update the node's current CH (location and energy)
        node.location = new_ch_location
        node.energy = new_ch_energy
        
        # Update the current CH index
        node.curr_CH = new_ch_index
    
    #All nodes have energy zero
    else:
        remove_fully_dead_cluster(node, nodes)


def plot_graph(all_nodes, all_paths, show_neighbors=True, show_child=True, interval=5):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Helper function to update the plot frame by frame
    def update(frame):
        ax.clear()  # Clear the previous frame
        if frame < len(all_nodes):
            nodes = all_nodes[frame]  # Get the current state of nodes
            # path1 = all_paths[frame]   # Get the current path
        else:
            return

        # Plot the edges (neighbors)
        if show_neighbors:
            for node in nodes:
                x, y = node.location  # Get the (x, y) coordinates of the current node
                for neighbor_id in node.neighbors:
                    neighbor = nodes[neighbor_id]  # Get the neighbor node
                    x_n, y_n = neighbor.location   # Neighbor's coordinates
                    # Plot edge between the node and its neighbor
                    ax.plot([x, x_n], [y, y_n], color='black', linestyle='-', linewidth=1)

        # Plot the child nodes (pink edges and purple child nodes)
        if show_child:
            for node in nodes:
                x, y = node.location
                for child in node.cluster_locations:
                    x_n, y_n = child   # Child's coordinates
                    # Plot edge between the node and its child
                    ax.plot([x, x_n], [y, y_n], color='pink', linestyle='-', linewidth=1)
            
            for node in nodes:
                for index, child in enumerate (node.cluster_locations):
                    x, y = child   # Child's coordinates
                    plt.scatter(x, y, color='purple', s=50)  # Red for all other nodes
                    plt.text(x+1 , y+1, f'{node.cluster_energies[index]}', fontsize=8)

        # Plot the nodes
        for i, node in enumerate(nodes):
            x, y = node.location
            if i == 0:  # Source node
                ax.scatter(x, y, color='blue', s=100, label='Source' if i == 0 else "")  # Blue for source
            elif i == 1:  # Sink node
                ax.scatter(x, y, color='green', s=100, label='Sink' if i == 1 else "")  # Green for sink
            else:
                ax.scatter(x, y, color='red', s=100)  # Red for all other nodes
            # Add labels to the nodes (node_id)
            ax.text(x, y, f'{node.node_id}', fontsize=12)

        # # Plot the path1 in green
        # for k in range(len(path1) - 1):
        #     node1 = nodes[path1[k]]     # Current node
        #     node2 = nodes[path1[k + 1]] # Next node in the path
        #     x1, y1 = node1.location
        #     x2, y2 = node2.location
        #     # Plot a green edge between adjacent nodes in the path
        #     ax.plot([x1, x2], [y1, y2], color='green', linestyle='-', linewidth=2)

        # Set plot labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Graph of Nodes and Path - Frame {frame+1}/{len(all_nodes)}")

        # Show the legend for source and sink
        ax.legend()

        # Display grid
        ax.grid(True)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(all_nodes), interval=interval, repeat=False)

    # Show the animation
    plt.show()

# def remove_node(nodes, x, rho, tau_min, tau_max):
#     nodes[x].energy=0
#     for neighbor in nodes[x].neighbors:
#         nodes[0].pheromone_matrix[neighbor][x]=0
#         nodes[0].pheromone_matrix[x][neighbor]=0
#     for i in range(5):
#         establish_route(0, 1, nodes, rho, tau_min, tau_max)
    
#     # print("New path: ", generate_data_path(nodes[0],1,nodes))

def priint(matrix):
    print("Pheromone Matrix:")
    for row in matrix:
        print([f"{value:.2f}" for value in row])

def priint_energies(nodes):
    for node in nodes:
        print(node.node_id, node.energy)
        

def plot_alive_nodes(data):
    """
    Plots the number of alive nodes versus the round number.

    Parameters:
    data (list of tuples): Each element is a tuple (round_number, alive_nodes).
    """
    # Unzip the data into two lists: rounds and alive_nodes
    rounds, alive_nodes = zip(*data)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, alive_nodes, linestyle='-', color='b', label='Alive Nodes')
    
    # Add labels and title
    plt.xlabel('Round Number')
    plt.ylabel('Number of Alive Nodes')
    plt.title('Alive Nodes Over Rounds')
    
    # Add a grid for better readability
    plt.grid(True)
    
    # Show the legend
    plt.legend()
    
    # Show the plot
    plt.show()

def plot_graph_static(nodes, path1, path2=[], show_neighbors=True, show_child=True):
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

    if show_child:
        for node in nodes:
            x,y=node.location
            for child in node.cluster_locations:
                x_n, y_n = child   # Child's coordinates
                # Plot edge between the node and its neighbor
                plt.plot([x, x_n], [y, y_n], color='pink', linestyle='-', linewidth=1)
        
        for node in nodes:
            for index, child in enumerate (node.cluster_locations):
                x, y = child   # Child's coordinates
                plt.scatter(x, y, color='purple', s=50)  # Red for all other nodes
                plt.text(x+1 , y+1, f'{node.cluster_energies[index]}', fontsize=8)

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



def F_Disconnected_N(nodes, rho, tau_min, tau_max, Elec, epsilon):
    alive_nodes=[]
    sink_node=nodes[1]

    for node in nodes:
        if node.node_id != 1:
            for i in range(5):
                establish_route(node.node_id, sink_node.node_id, nodes, rho, tau_min, tau_max)
    loop=0
    mini=0.5
    total_alive=len(nodes)-1
    all_sent=[]
    while (total_alive>0):
        loop+=1
        total_alive=0
        for node in nodes:
            if node.node_id != 1 and node.energy>0:
                dropped, opti_path = send_data(node.node_id, sink_node.node_id, nodes, None, 4000, Elec, epsilon, rho, tau_min, tau_max)
                all_sent.append(opti_path)
                if(opti_path[-1]==1):
                    total_alive+=1
                # if (loop %1000 )==1 and loop>1000:
                #     # all_nodes.append(copy.deepcopy(nodes))
                #     for i in range(len(nodes)):
                #         print((all_sent[-(i+1)]))  
        
        # counter=0
        # total=0
        # for node in nodes:
        #     if node.node_id != 1:
        #         for energy in node.cluster_energies:
        #             mini=min(mini,energy)
        #             if energy>0:
        #                 counter+=1
        #         total+=counter
        
        if((loop%1000 )== 1):
            print("alive: ", total_alive, "loop: ", loop)
            priint_energies(nodes)
        alive_nodes.append((loop,total_alive))
    
    return alive_nodes



def FDN(nodes, rho, tau_min, tau_max, Elec, epsilon):
    alive_nodes=[]
    sink_node=nodes[1]

    for node in nodes:
        if node.node_id != 1:
            for i in range(5):
                establish_route(node.node_id, sink_node.node_id, nodes, rho, tau_min, tau_max)
    loop=0
    mini=0.5
    total=len(nodes)
    all_sent=[]
    while (total>0 and loop<5000):
        loop+=1
        for node in nodes:
            if node.node_id != 1 and node.energy>0:
                dropped, opti_path = send_data(node.node_id, sink_node.node_id, nodes, None, 4000, Elec, epsilon, rho, tau_min, tau_max)
                all_sent.append(opti_path)
                if (loop %1000 )==1 and loop>1000:
                    # all_nodes.append(copy.deepcopy(nodes))
                    for i in range(len(nodes)):
                        print((all_sent[-(i+1)]))  
        
        counter=0
        total=0
        for node in nodes:
            if node.node_id != 1:
                for energy in node.cluster_energies:
                    mini=min(mini,energy)
                    if energy>0:
                        counter+=1
                total+=counter
        
        if((loop%1000 )== 1):
            print("alive: ", total, "loop: ", loop)
            priint_energies(nodes)
        alive_nodes.append((loop,counter))
    
    return alive_nodes


def with_erasure(nodes, rho,tau_min, tau_max, number_of_rounds=500):
    
    for node in nodes:
            if node.node_id != 1:
                for i in range(5):
                    establish_route(node.node_id, 1, nodes, rho, tau_min, tau_max)
    print("starting")
    loop=0
    success=0
    erasure_code=[]
    while(loop<number_of_rounds):
        loop+=1
        if(loop%100==1):
            print("Loop: ", loop)
        for node in nodes:
            if node.node_id != 1:
                dropped, opti_path = send_data(node.node_id, 1, nodes, 15, 400, Elec, epsilon, rho, tau_min, tau_max)
                if(dropped<=5):
                    success+=1
        erasure_code.append((loop, success))
    
    return erasure_code

def wo_erasure_code(nodes, rho,tau_min, tau_max, number_of_rounds=500):
    
    wo_erasure_code=[]
    loop=0
    success=0
    while(loop<number_of_rounds):
        loop+=1
        if(loop%100==1):
            print("Loop: ", loop)
        for node in nodes:
            if node.node_id != 1:
                dropped, opti_path = send_data(node.node_id, 1, nodes, 10, 400, Elec, epsilon, rho, tau_min, tau_max)
                if(dropped==0):
                    success+=1
        wo_erasure_code.append((loop, success))

    return wo_erasure_code

def packet_loss_ratio(nodes, rho, tau_min, tau_max, mini_loss, max_loss):
    #define np.matrix of size nXn with each entry being a random number between 0.3 to 0.4 and aij=aji
    loss_matrix=np.random.rand(len(nodes),len(nodes)) 
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            loss_matrix[i][j]=loss_matrix[i][j]*(max_loss-mini_loss)+mini_loss
            loss_matrix[j][i]=loss_matrix[i][j]
    for node in nodes:
        node.loss_matrix=loss_matrix

    #deepcopy nodes
    nodes1=copy.deepcopy(nodes)
    number_of_rounds=500
    w_erasure=with_erasure(nodes, rho, tau_min, tau_max, number_of_rounds)
    wo_erasure=wo_erasure_code(nodes1, rho, tau_min, tau_max, number_of_rounds)
    plot_packet_loss(w_erasure, wo_erasure)
    print("Packet Success Ratio with Erasure Code: ", w_erasure[-1][1]/((len(nodes)-1)*number_of_rounds))
    print("Packet Success Ratio without Erasure Code: ", wo_erasure[-1][1]/((len(nodes)-1)*number_of_rounds))
    
def plot_packet_loss(w_erasure, wo_erasure):
    rounds1, success1 = zip(*w_erasure)
    rounds2, success2 = zip(*wo_erasure)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rounds1, success1, linestyle='-', color='b', label='With Erasure Code')
    plt.plot(rounds2, success2, linestyle='-', color='r', label='Without Erasure Code')
    
    plt.xlabel('Round Number')
    plt.ylabel('Number of Successful Packets')
    plt.title('Packet Loss Ratio Over Rounds')
    
    plt.grid(True)
    plt.legend()
    plt.show()


# Example of setting up the network, initializing ants, and moving ants
n = 20              # Number of nodes
side_length = 200    # Side length of square area of network
energy = 0.5        # Same energy level for all nodes
r_min = 40            # Neighbor node range
r_max = 100*(1)            # Neighbor node range
num_ants = 100        # Number of ants
initial_pheromone=10.0
rho = 0.1
tau_max = 40
tau_min = 2
Elec = 50*(10**(-9))
epsilon = 0.00131*10**(-12)


# Generate nodes
nodes = initialize_nodes(n,side_length, r_min, r_max,energy, num_ants, initial_pheromone)
main_source_node=nodes[0]
main_sink_node=nodes[1]
path=[]

plot_graph_static(nodes, path)

# all_paths = []
# all_nodes = []

    
packet_loss_ratio(nodes, rho, tau_min, tau_max, 0.1, 0.2)
# alive_nodes = FDN(nodes, rho, tau_min, tau_max, Elec, epsilon)
# alive_nodes = F_Disconnected_N(nodes, rho, tau_min, tau_max, Elec, epsilon)
# plot_graph(all_nodes,[], show_neighbors=False, interval=1)
# plot_alive_nodes(alive_nodes)


# for i in range(5):
#     establish_route(source_node.node_id, sink_node.node_id, nodes, rho, tau_min, tau_max)
#     # priint(source_node.pheromone_matrix)
#     path=generate_data_path(source_node, sink_node.node_id,nodes)
#     #print(path)
#     all_paths.append(path)  # Store the current path
#     all_nodes.append(copy.deepcopy(nodes))  
#     # plot_graph(nodes, path, show_neighbors=False)
# print("Original Path: ", path)
# lop=0
# mini=0.5
# while (abs(mini)>0 ):
# # for i in range(12000):
#     send_data(0,1,nodes,"",Elec,epsilon, rho, tau_min, tau_max) 
#     mini=0.5
#     lop+=1
#     for node in nodes:
#         mini=min(mini, node.energy)
# print("First dead node ", lop)
# priint_energies(nodes)
# path1=path
# for i in range(5):
#     establish_route(source_node.node_id, sink_node.node_id, nodes, rho, tau_min, tau_max)
#     # priint(source_node.pheromone_matrix)
#     path=generate_data_path(source_node, sink_node.node_id,nodes)
#     #print(path)
#     # plot_graph(nodes, path, show_neighbors= False)

# plot_graph(all_nodes, all_paths, show_neighbors=False, interval=0.1)

# # priint_energies(nodes)
