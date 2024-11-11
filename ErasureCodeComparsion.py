import time
import random
from reedsolo import RSCodec
from zfec import Encoder, Decoder
import math

def encode_decode_reed_solomon(data, parity_symbols, simulate_loss=False):
    rsc = RSCodec(parity_symbols)
    start_time = time.time_ns()
    encoded_data = rsc.encode(data)
    encoding_time = (time.time_ns() - start_time) / 1_000_000

    if simulate_loss:
        corrupted_data = bytearray(encoded_data)
        indices_to_remove = random.sample(range(len(corrupted_data)), parity_symbols)
        corrupted_data = [b if i not in indices_to_remove else None for i, b in enumerate(corrupted_data)]
        corrupted_data = bytes([b for b in corrupted_data if b is not None])
    else:
        indices_to_corrupt = random.sample(range(len(encoded_data)), parity_symbols)
        corrupted_data = bytearray(encoded_data)
        for index in indices_to_corrupt:
            corrupted_data[index] ^= 0xFF

    start_time = time.time_ns()
    try:
        reconstructed_data = rsc.decode(corrupted_data)[0]
        success = reconstructed_data == data
    except Exception:
        reconstructed_data = None
        success = False
    decoding_time = (time.time_ns() - start_time) / 1_000_000

    return encoding_time, decoding_time, success

def encode_decode_cauchy(data, data_blocks, parity_blocks):
    total_blocks = data_blocks + parity_blocks
    block_size = math.ceil(len(data) / data_blocks)
    padded_data = data.ljust(block_size * data_blocks, b'\x00')
    data_blocks_list = [padded_data[i * block_size:(i + 1) * block_size] for i in range(data_blocks)]

    encoder = Encoder(data_blocks, total_blocks)
    start_time = time.time_ns()
    encoded_blocks = encoder.encode(data_blocks_list)
    encoding_time = (time.time_ns() - start_time) / 1_000_000

    indices_to_remove = random.sample(range(total_blocks), parity_blocks)
    received_blocks = [block for i, block in enumerate(encoded_blocks) if i not in indices_to_remove]
    received_indices = [i for i in range(total_blocks) if i not in indices_to_remove]

    if len(received_blocks) < data_blocks:
        print("Not enough uncorrupted blocks to reconstruct data.")
        return encoding_time, None, False

    decoder = Decoder(data_blocks, total_blocks)
    start_time = time.time_ns()
    try:
        reconstructed_blocks = decoder.decode(received_blocks, received_indices)
        reconstructed_data = b''.join(reconstructed_blocks)[:len(data)]
        success = reconstructed_data == data
    except Exception:
        reconstructed_data = None
        success = False
    decoding_time = (time.time_ns() - start_time) / 1_000_000

    return encoding_time, decoding_time, success

if __name__ == "__main__":
    with open('input40kB.txt', 'rb') as f:
        data = f.read()

    file_size = len(data)
    print(f"File size: {round(file_size / 1024)} kB")
    
    data_blocks = round(file_size/400)
    parity_blocks = round(data_blocks/10)

    if parity_blocks < 1:
        print("File size is less than 10KB. No erasure coding performed.")
        exit()


    print("Parameters:")
    print(f"  Data blocks (k): {data_blocks}")
    print(f"  Parity blocks (m): {parity_blocks}")
    print(f"  Total blocks (n): {data_blocks + parity_blocks}")

    print("Standard Reed-Solomon:")
    rs_enc_time, rs_dec_time, rs_success = encode_decode_reed_solomon(data, parity_blocks)
    print(f"  Encoding time: {rs_enc_time:.4f} ms")
    print(f"  Decoding time: {rs_dec_time:.4f} ms")
    print(f"  Reconstruction {'succeeded' if rs_success else 'failed'}")

    print("Cauchy Reed-Solomon:")
    cauchy_enc_time, cauchy_dec_time, cauchy_success = encode_decode_cauchy(data, data_blocks, parity_blocks)
    print(f"  Encoding time: {cauchy_enc_time:.4f} ms")
    if cauchy_dec_time is not None:
        print(f"  Decoding time: {cauchy_dec_time:.4f} ms")
    else:
        print("  Decoding failed due to insufficient uncorrupted blocks.")
    print(f"  Reconstruction {'succeeded' if cauchy_success else 'failed'}")
