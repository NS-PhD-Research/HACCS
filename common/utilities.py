import socket
import hashlib
from random import randint

class Utility:
    def __init__(self) -> None:
        pass

    def get_free_port():
        while True:
            port = randint(32768, 61000)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if not (sock.connect_ex(('127.0.0.1', port)) == 0):
                return port
    get_free_port = staticmethod(get_free_port)
    
    def get_id(msg):
        md5 = hashlib.md5()
        md5.update(msg.encode())
        return str(int(md5.hexdigest(), 16))[0:12]
    get_id = staticmethod(get_id) 