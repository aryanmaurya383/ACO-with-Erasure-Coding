import time
import random
from reedsolo import RSCodec
from zfec import Encoder, Decoder
import math

def encode_decode_reed_solomon(data, parity_symbols, simulate_loss=False):
    # Initialize counters
    xor_count_enc = 0
    addition_count_enc = 0
    multiplication_count_enc = 0

    # Encoding
    rsc = RSCodec(parity_symbols)
    encoded_data = rsc.encode(data)
    xor_count_enc += len(data) * parity_symbols
    addition_count_enc += len(data) * parity_symbols
    multiplication_count_enc += len(data) * parity_symbols

    enc_energy = calculate_energy(xor_count_enc, addition_count_enc, multiplication_count_enc)
    print("\nReed-Solomon Encoding:")
    print(f"  XORs: {xor_count_enc}, Additions: {addition_count_enc}, Multiplications: {multiplication_count_enc}")
    print(f"  Energy used: {enc_energy:.6f} µJ")

    # Simulate data corruption
    if simulate_loss:
        corrupted_data = bytearray(encoded_data)
        indices_to_remove = random.sample(range(len(corrupted_data)), parity_symbols)
        corrupted_data = [b if i not in indices_to_remove else None for i, b in enumerate(corrupted_data)]
        corrupted_data = bytes([b for b in corrupted_data if b is not None])
    else:
        corrupted_data = bytearray(encoded_data)
        for index in random.sample(range(len(encoded_data)), parity_symbols):
            corrupted_data[index] ^= 0xFF

    # Decoding
    xor_count_dec = 0
    addition_count_dec = 0
    multiplication_count_dec = 0
    try:
        reconstructed_data = rsc.decode(corrupted_data)[0]
        xor_count_dec += len(corrupted_data) * parity_symbols
        addition_count_dec += len(corrupted_data) * parity_symbols
        multiplication_count_dec += len(corrupted_data) * parity_symbols
        success = reconstructed_data == data
    except Exception:
        reconstructed_data = None
        success = False

    dec_energy = calculate_energy(xor_count_dec, addition_count_dec, multiplication_count_dec)
    print("\nReed-Solomon Decoding:")
    print(f"  XORs: {xor_count_dec}, Additions: {addition_count_dec}, Multiplications: {multiplication_count_dec}")
    print(f"  Energy used: {dec_energy:.6f} µJ")
    print(f"  Success: {'Yes' if success else 'No'}")

    return success


def encode_decode_cauchy(data, data_blocks, parity_blocks):    
    # Block details
    total_blocks = data_blocks + parity_blocks
    block_size = math.ceil(len(data) / data_blocks)
    padded_data = data.ljust(block_size * data_blocks, b'\x00')
    data_blocks_list = [padded_data[i * block_size:(i + 1) * block_size] for i in range(data_blocks)]

    # Encoding
    encoder = Encoder(data_blocks, total_blocks)
    xor_count_enc = 0
    addition_count_enc = 0
    multiplication_count_enc = 0
    encoded_blocks = encoder.encode(data_blocks_list)
    xor_count_enc += data_blocks * parity_blocks * block_size

    enc_energy = calculate_energy(xor_count_enc, addition_count_enc, multiplication_count_enc)
    print("\nCauchy Reed-Solomon Encoding:")
    print(f"  XORs: {xor_count_enc}, Additions: {addition_count_enc}, Multiplications: {multiplication_count_enc}")
    print(f"  Energy used: {enc_energy:.6f} µJ")

    # Simulate data corruption
    indices_to_remove = random.sample(range(total_blocks), parity_blocks)
    received_blocks = [block for i, block in enumerate(encoded_blocks) if i not in indices_to_remove]
    received_indices = [i for i in range(total_blocks) if i not in indices_to_remove]

    if len(received_blocks) < data_blocks:
        print("\nCauchy Reed-Solomon Decoding:")
        print("  Insufficient blocks to recover data.")
        return False

    # Decoding
    decoder = Decoder(data_blocks, total_blocks)
    xor_count_dec = 0
    addition_count_dec = 0
    multiplication_count_dec = 0
    try:
        reconstructed_blocks = decoder.decode(received_blocks, received_indices)
        xor_count_dec += data_blocks * parity_blocks * block_size
        reconstructed_data = b''.join(reconstructed_blocks)[:len(data)]
        success = reconstructed_data == data
    except Exception:
        reconstructed_data = None
        success = False

    dec_energy = calculate_energy(xor_count_dec, addition_count_dec, multiplication_count_dec)
    print("\nCauchy Reed-Solomon Decoding:")
    print(f"  XORs: {xor_count_dec}, Additions: {addition_count_dec}, Multiplications: {multiplication_count_dec}")
    print(f"  Energy used: {dec_energy:.6f} µJ")
    print(f"  Success: {'Yes' if success else 'No'}")

    return success


def calculate_energy(xor_count, addition_count, multiplication_count):
    energy_per_xor = 0.1e-12
    energy_per_addition = 0.1e-12
    energy_per_multiplication = 3e-12

    total_energy = (xor_count * energy_per_xor +
                    addition_count * energy_per_addition +
                    multiplication_count * energy_per_multiplication)

    return total_energy * 1e6


if __name__ == "__main__":
    import random

    with open('input40kB.txt', 'rb') as f:
        data = f.read()

    file_size = len(data)
    print(f"File size: {file_size / 1024:.2f} kB")

    data_blocks = max(round(file_size / 400), 1)
    parity_blocks = max(round(data_blocks / 5), 1)

    print("\nConfiguration:")
    print(f"  Data blocks: {data_blocks}")
    print(f"  Parity blocks: {parity_blocks}")
    print(f"  Total blocks: {data_blocks + parity_blocks}")

    encode_decode_reed_solomon(data, parity_blocks)
    encode_decode_cauchy(data, data_blocks, parity_blocks)
