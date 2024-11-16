# **EC-ACO with Erasure Coding**

## **Project Overview**
This project implements the EC-ACO algorithm and compares different erasure coding techniques, including Cauchy Reed-Solomon (CRS) and Reed-Solomon (RS) coding. It uses Python libraries to perform simulations, visualize results, and evaluate the performance of the proposed approach.

---

## **File and Folder Descriptions**
ACO_all_nodes.py: Contains the implementation of the EC-ACO algorithm.
Erasure_Code.py: Provides a comparison between Cauchy Reed-Solomon (CRS) and Reed-Solomon (RS) erasure coding techniques.
Documents: Contains presentation and report

## **Required Libraries**
The following Python libraries are required to run the project:

1. **NumPy**: For numerical computations.
2. **Matplotlib**: For plotting and visualization.
3. **networkx**: For graph/network generation and analysis.
4. **reedsolo**: For Reed-Solomon error correction.
5. **zfec**: For erasure coding and decoding (Encoder and Decoder).

---

## **Installation**
You can install all the required libraries using the following pip commands:

```bash
pip install numpy matplotlib networkx reedsolo zfec
```
---

## **Running the project**
Ensure python and all required libraries are installed using the provided pip command.

Run the Python files to execute the EC-ACO and erasure coding files:

```bash
python ACO_all_nodes.py
python Erasure_Code.py
```