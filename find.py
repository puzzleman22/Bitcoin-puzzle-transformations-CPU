import secp256k1 as ice
import multiprocessing
import random
import secrets
import os
import time
import psutil
import math
import hashlib
import uuid
import base64
import zlib
import platform
import string
# Set current process to high priority
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)


def shuffle_string(s):
    char_list = list(s)
    random.shuffle(char_list)
    return ''.join(char_list)
    
def rotate_hex(hex_string):
    # Precompute a translation table for all hex digits
    translation_table = str.maketrans("0123456789abcdef", "123456789abcdef0")
    return hex_string.translate(translation_table)

def shift_left(s, n):
    n = n % len(s)
    return s[n:] + s[:n]
def inverse(binary_string):
    # Ensure the input is valid
    if not all(char in '01' for char in binary_string):
        raise ValueError("Input string must contain only '0' and '1'")
    
    return ''.join('1' if char == '0' else '0' for char in binary_string)


def run(process_id):
    size = 20
    hexSize = size // 4
    g = 0 
    ice.Load_data_to_memory('bloombtc.bin', True)
    #addrs = set()
    while True:

        bin2 = bin(secrets.randbits(size))[2:].zfill(size)[:size]
        for inv in range(2):
            for z in range(2):
                for y in range(size):
                    pp = int(bin2, 2)
                    hex2 = hex(pp)[2:].zfill(hexSize)
                    for x in range(16):
                        p = int('1' + hex2, 16)
                        h160 = ice.privatekey_to_h160(0, True, p)

                        if y == 0 and z == 0 and inv == 0 and x == 0:
                            print(str(g) + ' - ' + bin2 + ' - ' + hex(p)[2:] + ' -> ' + h160.hex())
                            
                        if ice.check_collision(h160): 
                            print(hex(p)[2:] + ' -> ' + h160.hex())
                            print('found')
                            with open('found.txt', 'a') as file:
                                file.write(hex(p)[2:] + ' -> ' +  h160.hex() + "\n")
                            return
                        hex2 = rotate_hex(hex2)
                    bin2 = shift_left(bin2, 1)
                bin2 = bin2[::-1]
            bin2 = inverse(bin2)
        g += 1 
# --- Entry Point ---

if __name__ == '__main__':
    processes = []
    for i in range(1):
        p = multiprocessing.Process(target=run, args=(i,))
        processes.append(p)
        p.start()
