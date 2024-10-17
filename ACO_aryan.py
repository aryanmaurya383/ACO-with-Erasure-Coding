import numpy as np
import random
import math

# Node Class
class Node:
    def __init__(self, node_id, x, y, energy, n, max_num_ants):
        self.node_id = node_id                      # ID of the node
        self.location = (x,y)                       # (x,y) location of the node
        self.energy = energy                        # Energy of the node
        self.neighbors = []                         # List of neighbors (node ids)
        self.num_neighbors = 0                      # Number of neighbors
        self.pheromone_matrix = np.zeros((n, n))    # Pheromone matrix (n x n)
        self.loss_matrix = np.zeros((n, n))         # Loss percentage matrix (n x n)
        self.max_num_ants = max_num_ants            # Max number of ants to be produced
        self.generated_ants_count = 0               # Number of ants generated
        self.ants = []                              # List of ants currently at this node

    def add_neighbor(self, neighbor_id, pheromone_level, loss_percent):
        self.neighbors.append(neighbor_id)
        self.pheromone_matrix[self.node_id][neighbor_id] = pheromone_level
        self.loss_matrix[self.node_id][neighbor_id] = loss_percent
        self.num_neighbors = len(self.neighbors)

    def initialize_k_ants(self):
        """
        Initializes the node by resetting the list of ants for the next iteration.
        Clears all previously stored ants.
        """
        self.ants = []  # Clear the ants list
        self.generated_ants_count = 0  # Reset the generated ants count
        print(f"Node {self.node_id}: Initialized for new ants.")

    def generate_ants(self, source_node, sink_node, pheromone_matrix):
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
            new_ant = Ant(ant_id, source_node, sink_node, pheromone_matrix)
            
            # Add the ant to the list of ants at this node
            self.ants.append(new_ant)
            
            # Increment the count of generated ants
            self.generated_ants_count += 1
            
            print(f"Ant {new_ant.ant_id} generated at Node {self.node_id}.")
        else:
            print(f"Node {self.node_id} has reached the max number of ants.")



# Ant Class
class Ant:
    def __init__(self, ant_id, source_node, sink_node, pheromone_matrix):
        self.ant_id = ant_id                        # Unique ID of the ant
        self.source_node = source_node              # Source node of the ant
        self.sink_node = sink_node                  # Destination node (sink)
        self.visited_nodes = [source_node]          # List of nodes visited, starting from the source
        self.current_position = source_node         # Current position (initially at source)
        self.distance_traveled = 0.0                # Total distance traveled
        self.curr_back_index=-1                     # Used in back ant how much path retraced
        self.is_back_ant = False                    # Indicates if the ant is retracing its path (True if retracing)
        self.pheromone_matrix = [row[:] for row in pheromone_matrix]  # Copy of the source node's pheromone matrix


    # Function to move the ant to the next hop
    def move_ant(self, nodes, alpha=1.0, beta=2.0, m=0.5, rho=0.1, Q=1.0, tau_min=0.1, tau_max=5.0):
        if self.is_back_ant:
            # Moving backward by using the index and updating the distance list
            if self.curr_back_index > 0:  # Ensure there's still a path to backtrack            
                # Update distance traveled by the distance between current and previous node
                prev_node = self.visited_nodes[self.curr_back_index - 1]
                curr_node = self.visited_nodes[self.curr_back_index]
                self.distance_traveled += euclidean_distance(nodes[curr_node], nodes[prev_node])
                
                # Move to the previous node in the visited path
                self.curr_back_index -= 1
                self.current_position = self.visited_nodes[self.curr_back_index]
                
                print(f"Back Ant {self.ant_id} moved to node {self.current_position}, total distance: {self.distance_traveled}")
            
            else:
                # If current back index is 0, it means we've reached the source node
                print(f"Ant {self.ant_id} has reached the source node while retracing.")
                store_ant(self, nodes[self.source_node])  # Store ant at the source node
                return  # Ant has finished its journey back to the source
            
            #If all ants reach the source node then update pheromone and remove ants
            if len(nodes[self.source_node].ants) == nodes[self.source_node].max_num_ants:
                # Evaporate pheromones and add pheromone from ants
                evaporate_pheromones(self.source_node, nodes)
                add_pheromone_from_ants(nodes[self.source_node], Q=1.0)

                # Initialize ants for the next round
                nodes[self.source_node].initialize_k_ants()

        else:
            # Calculate next hop using probabilities during forward movement
            current_node = nodes[self.current_position]
            probabilities = get_next_hop_probabilities(current_node, self, nodes, alpha, beta, m)
            next_hop = choose_next_hop(probabilities)
            self.visited_nodes.append(next_hop)

            if next_hop == self.sink_node:
                self.start_backtracking()
                print(f"Ant {self.ant_id} reached the Sink Node {self.sink_node}. Backtracking started.")

        current_node = nodes[self.current_position]
        self.distance_traveled += euclidean_distance(current_node, nodes[next_hop])
        self.current_position = next_hop

    def start_backtracking(self):
        self.is_back_ant = True
        self.curr_back_index=len(self.visited_nodes)-1
        print(f"Ant {self.ant_id} is now retracing its path back to the source.")


def store_ant(ant, source_node):
    """
    Stores the ant in the source node after it completes its journey back.
    """
    source_node.ants.append(ant)
    print(f"Ant {ant.ant_id} has been stored at the source node {source_node.node_id}.")


# Function to evaporate pheromones for all neighbors of a node, excluding visited nodes
def evaporate_pheromones(source_node, rho=0.1, tau_min=0.1):
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

            # Print updated pheromone information for debugging
            print(f"Pheromone evaporated between Node {i} and Node {j}: {tau_ij} -> {new_tau_ij}")

def add_pheromone_from_ants(source_node, Q=1.0, tau_max=5.0):
    """
    Adds pheromones to the source node's pheromone matrix based on the paths taken by the ants.
    For each ant, the function iterates through its visited nodes and adds Q / Lk to the pheromone
    levels between consecutive nodes in the path, ensuring the pheromone level does not exceed tau_max.

    :param source_node: The node that stores the pheromone matrix (source node).
    :param Q: Constant value used to calculate the pheromone contribution (default is 1.0).
    :param tau_max: Maximum allowed pheromone level (upper bound).
    """
    ants = source_node.ants
    for ant in ants:
        # Ensure the ant has a valid path and has completed its journey
        if len(ant.visited_nodes) < 2 or ant.distance_traveled <= 0:
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

            # Print debug information for the pheromone update
            print(f"Pheromone added between Node {node_i} and Node {node_j}: +{pheromone_contribution}, capped at {tau_max} if exceeded.")


# Function to update pheromone levels between two nodes using the given equation
# Modified function to update pheromones with evaporation, excluding visited nodes
def update_pheromone_backtracking(current_node_id, next_node_id, nodes, ant, rho=0.1, Q=1.0, tau_min=0.1, tau_max=5.0):
    """
    Updates pheromone levels during backtracking, performing evaporation on all neighboring
    links (excluding visited nodes) and updating the pheromone only on the current path.

    :param current_node_id: ID of the current node.
    :param next_node_id: ID of the next node the ant will visit in its retraced path.
    :param nodes: List of all nodes in the network.
    :param ant: The ant retracing its path.
    :param rho: Evaporation rate.
    :param Q: Constant used for pheromone increase.
    :param tau_min: Minimum pheromone level.
    :param tau_max: Maximum pheromone level.
    """
    current_node = nodes[current_node_id]

    # Evaporate pheromones on all neighbors except those in the visited path
    evaporate_pheromones(current_node_id, next_node_id, nodes, ant.visited_nodes, rho, tau_min)

    # Now update pheromone for the current path (i.e., the link between current_node and next_node)
    tau_ij = current_node.pheromone_matrix[current_node_id][next_node_id]
    delta_tau_ij = Q / ant.distance_traveled  # Calculate Δτ_ij(t) = Q / L_k

    # Apply the pheromone update equation
    new_tau_ij = (1 - rho) * tau_ij + delta_tau_ij

    # Ensure the pheromone level stays within the bounds
    if new_tau_ij > tau_max:
        new_tau_ij = tau_max
    elif new_tau_ij < tau_min:
        new_tau_ij = tau_min

    # Update the pheromone in both directions (symmetric)
    current_node.pheromone_matrix[current_node_id][next_node_id] = new_tau_ij
    nodes[next_node_id].pheromone_matrix[next_node_id][current_node_id] = new_tau_ij

    print(f"Pheromone updated between Node {current_node_id} and Node {next_node_id}: {tau_ij} -> {new_tau_ij}")



# Function to calculate Euclidean distance between two nodes
def euclidean_distance(node1, node2):
    return math.sqrt((node1.location[0] - node2.location[0]) ** 2 + (node1.location[1] - node2.location[1]) ** 2)


# Function to generate n nodes randomly in a 10x10 area and set neighbors based on distance
def initialize_nodes(n, side_length, r, initial_energy, max_num_ants=5):
    """
    Initializes the nodes with unique random locations within a square of side_length x side_length,
    ensuring no two nodes are at the same position.
    """
    nodes = []
    positions = set()

    while len(nodes) < n - 1:  # Leave one position for sink node
        x, y = round(random.uniform(0, side_length), 2), round(random.uniform(0, side_length), 2)
        if (x, y) not in positions:
            positions.add((x, y))
            new_node = Node(node_id=len(nodes), x=x, y=y, energy=initial_energy, n=n, max_num_ants=max_num_ants)
            nodes.append(new_node)

    # Adding the sink node at position (10, 10)
    sink_node = Node(node_id=n - 1, x=side_length, y=side_length, energy=initial_energy, n=n, max_num_ants=max_num_ants)  # Infinite energy for sink
    nodes.append(sink_node)

    # Now calculate neighbors based on communication radius r
    for node in nodes:
        for other_node in nodes:
            if node.node_id != other_node.node_id:
                distance = euclidean_distance(node, other_node)
                if distance <= r:
                    node.neighbors.append(other_node.node_id)

    return nodes


# Function to initialize ants with a given source node and sink node
def initialize_ants(num_ants, source_node, nodes, sink_node):
    ants = []
    
    for i in range(num_ants):
        ant = Ant(i, source_node, sink_node, nodes[source_node].pheromone_matrix )  # Assign a unique ant ID
        ants.append(ant)
    
    return ants


# Function to calculate the eta value for a given neighbor
def get_eta(current_node, neighbor_node, sink_node, m=0.5):
    """
    Calculate the eta value for a neighbor.
    
    :param current_node: The node where the ant is currently located.
    :param neighbor_node: The neighbor node being evaluated.
    :param sink_node: The sink node (destination of the ant).
    :param m: Weight factor for distance calculation.
    
    :return: The eta value based on distance, energy, and number of neighbors.
    """
    dist_i_j = euclidean_distance(current_node, neighbor_node)
    dist_j_sink = euclidean_distance(neighbor_node, sink_node)
    
    energy_ratio = neighbor_node.energy / current_node.energy
    max_neighbors = max([node.num_neighbors for node in nodes])  # Max neighbors across all nodes
    
    eta = (1 / (m * dist_i_j + (1 - m) * dist_j_sink)) * (energy_ratio) * (neighbor_node.num_neighbors / max_neighbors)
    
    return eta


# Function to calculate the probability list for choosing the next hop
def get_next_hop_probabilities(node, ant, nodes, alpha=1.0, beta=2.0, m=0.5):
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
    sink_node = nodes[ant.sink_node]  # Sink node
    valid_neighbors = []  # List of neighbors that the ant can visit (not already visited)

    # Collect the pheromones and eta values for neighbors
    for neighbor_id in node.neighbors:
        if neighbor_id not in ant.visited_nodes:  # Avoid returning to visited nodes
            neighbor_node = nodes[neighbor_id]
            pheromone_level = node.pheromone_matrix[current_id][neighbor_id]
            eta_value = get_eta(node, neighbor_node, sink_node, m)
            
            pheromones.append(pheromone_level ** alpha)  # Pheromone influence
            etas.append(eta_value ** beta)               # Eta influence
            valid_neighbors.append(neighbor_id)          # Add valid neighbor ID
        else:
            pheromones.append(0)  # No pheromone for visited nodes
            etas.append(0)        # No eta for visited nodes

    # Compute the product of pheromone influence and eta for each neighbor
    combined_influences = [pheromones[i] * etas[i] for i in range(len(pheromones))]

    total_influence = sum(combined_influences)

    # If the total influence is zero, return zero probabilities
    if total_influence == 0:
        return [(neighbor_id, 0) for neighbor_id in valid_neighbors]
    
    # Calculate the probability for each neighbor
    probabilities = [(valid_neighbors[i], combined_influences[i] / total_influence) for i in range(len(valid_neighbors))]

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


def establish_route(source_node_id, sink_node_id, k_ants, nodes, moves_per_generation=3):
    """
    Establish a route by generating k ants from source to sink and back.
    Move ants and generate new ants after every few steps until all k ants are generated.
    
    :param source_node_id: ID of the source node
    :param sink_node_id: ID of the sink node
    :param k_ants: Number of ants to generate
    :param nodes: List of all nodes in the network
    :param moves_per_generation: Number of moves each ant takes before new ants are generated
    """
    ants = []
    total_ants_generated = 0

    while total_ants_generated < k_ants or len(ants) > 0:
        # Generate ants if we haven't reached the target k ants
        if total_ants_generated < k_ants:
            new_ant = Ant(ant_id=total_ants_generated, source_node=source_node_id, sink_node=sink_node_id)
            ants.append(new_ant)
            total_ants_generated += 1
            print(f"Generated Ant {new_ant.ant_id} at Node {source_node_id}")

        # Move each ant for a few steps (moves_per_generation times)
        for ant in ants:
            for _ in range(moves_per_generation):
                if ant.is_back_ant and len(ant.visited_nodes) == 1:
                    print(f"Ant {ant.ant_id} reached the source node and died.")
                    ants.remove(ant)
                    break
                ant.move_ant(nodes)

        # Check if all ants have returned to the source node
        ants = [ant for ant in ants if not (ant.is_back_ant and len(ant.visited_nodes) == 1)]

def send_data(source_node_id, sink_node_id, nodes, data):
    """
    Sends a data packet from source to sink. Uses pheromone levels to decide the next hop
    while considering the packet loss ratio from the loss matrix of the current node.
    
    :param source_node_id: ID of the source node
    :param sink_node_id: ID of the sink node
    :param nodes: List of all nodes in the network
    :param data: The data to be sent in the packet
    """
    sent_packets = []
    received_packets = []
    
    # Initialize the data packet with source, sink, current position, path, and the data field.
    packet = {
        'source': source_node_id,
        'sink': sink_node_id,
        'current_position': source_node_id,
        'path': [source_node_id],
        'data': data,
        'lost': False
    }
    sent_packets.append(packet)
    print(f"Data packet sent from Source Node {source_node_id} to Sink Node {sink_node_id} with data: {data}")

    # Move the packet through the network
    while packet['current_position'] != sink_node_id:
        next_hop = move_data(packet, nodes)
        if next_hop is None:  # Packet dropped due to loss
            packet['lost'] = True
            print(f"Packet dropped while moving from Node {packet['current_position']}.")
            return sent_packets, received_packets

        # Update the packet's path and current position
        packet['path'].append(next_hop)
        packet['current_position'] = next_hop

    # If it reaches the sink, add it to the received packets list
    print(f"Packet successfully reached Sink Node {sink_node_id} with data: {packet['data']}")
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
    max_pheromone = -1
    next_hop = None

    # Find the neighbor with the highest pheromone level (considering the loss ratio)
    for neighbor_id in current_node.neighbors:
        loss_ratio = current_node.loss_matrix[current_node.node_id][neighbor_id]
        pheromone_level = current_node.pheromone_matrix[current_node.node_id][neighbor_id]

        # Packet can only be sent if the random value exceeds the loss ratio
        if pheromone_level > max_pheromone and random.random() > loss_ratio:
            max_pheromone = pheromone_level
            next_hop = neighbor_id

    if next_hop is not None:
        print(f"Moving packet to Node {next_hop} from Node {packet['current_position']}.")
        return next_hop  # Return the next node to which the packet should move
    else:
        print(f"Packet lost due to high loss ratio from Node {packet['current_position']}.")
        return None  # Packet dropped due to loss


# Example of setting up the network, initializing ants, and moving ants
n = 30              # Number of nodes
side_length = 10    # Side length of square area of network
energy = 1.0        # Same energy level for all nodes
r = 3.0             # Communication range
num_ants = 5        # Number of ants
source_node = 0     # Source node for the ants
sink_node = 29      # Sink node (destination) for the ants

# Generate nodes
nodes = initialize_nodes(n,side_length, r,energy)

# Initialize ants
ants = initialize_ants(num_ants, source_node, nodes, sink_node)

# Move the first ant
first_ant = ants[0]
print(f"Ant {first_ant.ant_id} starting at Node {first_ant.source_node}")

# Move the
