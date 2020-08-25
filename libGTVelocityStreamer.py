import socket
import struct
import time
import numpy as np
from datetime import datetime

import logging



class GTVelocityStreamer:
    def __init__(self, ip, port, timeout=0.01):

        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.socket = None

        self.dataformat = 'dffffff'
        self.datamaxsize = 128

        self.alive = False

        

        self.open()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32)      

    def open(self):
        # declare our serverSocket upon which
        # we will be listening for UDP messages
        serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # One difference is that we will have to bind our declared IP address
        # and port number to our newly declared serverSock
        serverSock.bind((self.ip, self.port))
        # if self.timeout != 0:
        serverSock.settimeout(self.timeout)
        self.socket = serverSock

        return serverSock

    def close(self):
        self.socket.close()
    
    def read_data(self):
            
        try:
            # read all existing in the buffer and take only the last
            
            # self.socket.flush()

            buffer, addr = self.socket.recvfrom(self.datamaxsize)
            data = struct.unpack(self.dataformat, buffer) #< little endian, > big-endian
            # if len(self.dataformat)==6:
            #     data = tuple(a)
            data = tuple(list(data))
            self.alive = True
            return data
        #If data is not received back from server, print it has timed out  
        except socket.timeout:
            self.alive = False
            logging.debug("GTVelocityStreamer : Timeout. Is GTVelocityStreamer connected?")
        #     return None
        return None

    def read_data_last_msg(self):
        i = 0
        while 1:
            try:

                i = i+1
                # read all existing in the buffer and take only the last
                buffer, addr = self.socket.recvfrom(self.datamaxsize)
                data = struct.unpack(self.dataformat, buffer) #< little endian, > big-endian
                
            #If data is not received back from server, print it has timed out  
            except socket.timeout:
                print("timeout")
                break
            #     return None
            # if len(self.dataformat)==6:
            #     data = tuple(a)
        buffer, addr = self.socket.recvfrom(self.datamaxsize)
        data = struct.unpack(self.dataformat, buffer) #< little endian, > big-endian
        data = tuple(list(data))
        return data

    
    def read_position_velocity(self):
        # data = self.read_data_last_msg()
        data = self.read_data()
        if data is None:
            # return ((np.nan,np.nan,np.nan),(np.nan,np.nan,np.nan))
            return ((0,0,0),(0,0,0))
        t,x,y,z,vx,vy,vz = data
        return ((x,y,z), (vx, vy, vz))
            

    def speed(self):

        position, velocity = self.read_position_velocity()

        speed = np.linalg.norm(velocity) 
        # print(speed, position, velocity)
        return speed

    def check_alive(self):

        data = self.read_data()
        if data is None:
            return False
        t,x,y,z,vx,vy,vz = data
        # print(t)
        if t>=0:
            return True
        else:
            return False
