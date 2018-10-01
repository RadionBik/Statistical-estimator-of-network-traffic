#!/usr/bin/python3

'''
local side:
./packet_transceiver.py -d 192.168.88.20 -o obj/skypeLANhome_UDP_192_168_0_102_18826_auto_components.pkl -m local -i wlp3s0

./packet_transceiver.py -m local -d 172.17.0.3 -o obj/rtp_711_UDP_10_1_3_143_5000_auto_components.pkl -i eth0

remote side;
./packet_transceiver.py -d 192.168.88.10 -o obj/skypeLANhome_UDP_192_168_0_102_18826_auto_components.pkl -m remote

'''
import cProfile

import socket
import os
import time
import socketserver
import subprocess
import argparse
from helper_functions import *
import random

class serverHandler(socketserver.BaseRequestHandler):
    
    def handle(self):
        self.data = self.request.recv(1024)
        print(self.data)
        pass

class udpPacketHandler(socketserver.BaseRequestHandler):

    def handle(self):
        pass
        #data = self.request[0].strip()
        #socket = self.request[1]
        #print("{} wrote:".format(self.client_address[0]))
        #print(data)
        #socket.sendto(data.upper(), self.client_address)

def receive_TCP(ip, port):
    server = socketserver.TCPServer(('', port), serverHandler)
    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()

def receive_UDP(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
        try:
            server_sock.bind(('',port))
        except OSError:
            print('receiving socket is in use!')
        while True:
            data, addr = server_sock.recvfrom(2048)
    #        print('received UDP: ',data)


def send_TCP(ip, port, payloadLength):
    payload = ''.join(['1' for i in range(payloadLength)])
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_sock:
        client_sock.connect((ip, port))
        client_sock.send(bytes(payload,'utf8'))
    #client_sock.close()

def send_UDP(ip, port, payloadLength):
    #overhead for UDP is usually about 42 bytes. unicode symbol is 8 bits
    payload = ''.join(['1' for i in range(payloadLength)])
    #payload = payloadGeneral[:payloadLength] 
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_sock:
        client_sock.sendto(bytes(payload,'utf8'), (ip, port)) 


def wait_for_remote_sync():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', 7999))
        sock.listen(1)
        print('waiting for syncing with the remote side...')
        conn, addr = sock.accept()
        with conn:
            print('got sync from ', addr)
        #sock.shutdown(socket.SHUT_RDWR)
        #sock.close()

def sync_remote(ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        result = sock.connect_ex((ip, 7999))
        #if connected
        if result == 0:
            print('synced connection with the remote side')
            break
        time.sleep(0.2)

#@profile
def iteratePacketPerDevice(estimatedDeviceParams, trafficExtremeValues, direction):
    '''
    generatePacketIterator() returns properties of a single packet given the EM-estimations,
    the max and min values of the parameters for a specific device and direction
    '''
    #random.seed(a=88)

    #pr = cProfile.Profile()
    #pr.enable()

    generatedPacketParameters = {'pktLen': None,'IAT':None }
    for parameter in ['pktLen', 'IAT']:
        
        mix = estimatedDeviceParams[direction][parameter]
        if isinstance(mix, sklearn.mixture.gaussian_mixture.GaussianMixture):
            #print(parameter, direction, mix.means_)
            paramLimit = trafficExtremeValues[direction][parameter]
            genParam = paramLimit['max'] + 1

            #select the component number that we are drawing sample from (Multinoulli)

            component = np.random.choice( range(mix.n_components), p=mix.weights_)
            #new version, w/o nunpy, requires python 3.6!
            #component = random.choices( range(mix.n_components), weights=mix.weights_ )

            #generate parameter until it fits the limits
            while not ((genParam <= paramLimit['max']) and (genParam >= paramLimit['min'])):
                if parameter == 'pktLen':
                    #packet length in bytes must be integer
                    genParam = round(np.asscalar(random.gauss(mix.means_[component], np.sqrt(mix.covariances_[component]))))
                else:
                    genParam = np.asscalar(random.gauss(mix.means_[component], np.sqrt(mix.covariances_[component])) )

        #if there is a list, send its properties
        else:
            
            if len(estimatedDeviceParams[direction][parameter])>0:
                genParam = estimatedDeviceParams[direction][parameter].pop(0)
            else:
                break

        generatedPacketParameters[parameter] = genParam
                   
    #pr.disable()
    #pr.print_stats(sort='time')
    yield generatedPacketParameters

def generate_packets(trafficParameters, args, direction, sock, remotePort=None):

    '''
    how-accurate-is-pythons-time-sleep

    On linux if you need high accuracy you might want to look into using ctypes to call nanosleep() or clock_nanosleep(). 

    https://stackoverflow.com/questions/38319606/how-to-get-millisecond-and-microsecond-resolution-timestamps-in-python
    

    '''
    estimatedParameterEM, trafficExtremeValues = trafficParameters
    print('Started generating packets...')
    generateUntil = time.time() + int(args.l)
    #setup max payload
    payload = ''.join(['1' for i in range(1500)])
    
    iat = []
    while time.time() < generateUntil:
        for device in estimatedParameterEM:
            #print(device)
            packetGenerator = iteratePacketPerDevice(estimatedParameterEM[device], trafficExtremeValues[device], direction)
            packet = dict(next(packetGenerator))
            #until we have traffic saved as lists
            try:
                if packet['IAT'] > 0.001:
                    #set correction for the execution time ~2ms (varies, of course) (not now)
                    time.sleep(packet['IAT'])
            except TypeError:
                break 

            if remotePort is not None:
                sock.sendto(bytes(payload[:packet['pktLen']],'utf8'), (args.destination, remotePort))
            else:
                sock.send(bytes(payload[:packet['pktLen']],'utf8'))
        
            iat.append(packet['IAT'])

    #close TCP socket
    if remotePort is None:
        sock.close()

    save_obj(iat,direction+'_iat')


def generate_traffic(trafficParameters, args, remotePort, sock):


    if args.m == 'remote':
        direction = 'to'
        time.sleep(0.5)
    elif args.m == 'local':
        direction = 'from'
    else:
        exit('wrong side, see help')


    if args.protocol == 'TCP':
        generate_packets(trafficParameters, args, direction, sock)
        
    elif args.protocol == 'UDP':
        generate_packets(trafficParameters, args, direction, sock, remotePort)
    else:
        exit('wrong transport protocol, see help')
    

def sendMsg(args, remotePort, payloadLength=80):
    if args.protocol == 'UDP':
        send_UDP(args.destination, remotePort, payloadLength)
    elif args.protocol == 'TCP':
        send_TCP(args.destination, remotePort, payloadLength)    
    else:
        exit('wrong transport protocol, see help')

#make several processes
#https://www.tutorialspoint.com/concurrency_in_python/concurrency_in_python_quick_guide.htm

#https://docs.python.org/3/library/socketserver.html

def receiveTraffic(args, localPort):

    if args.protocol == 'UDP':
        receive_UDP(args.source, localPort)
    elif args.protocol == 'TCP':
        receive_TCP(args.source, localPort)
    else:
        exit('wrong transport protocol, see help')


def get_first_datagram_port(sock):
    while True:
        data, addr = sock.recvfrom(1500)
        print('received first UDP from: ',addr)
        break
    return addr

def init_local_socket(args, LISTEN_PORT):

    if args.protocol == 'UDP':
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    elif args.protocol == 'TCP':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            result = sock.connect_ex((args.destination, LISTEN_PORT))
            #if connected
            if result == 0:
                print('connected to the remote side')
                break
            time.sleep(0.1)
    else:
        exit('wrong transport protocol, see help')

    return sock

def init_remote_socket(args, LISTEN_PORT):

    if args.protocol == 'UDP':
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('',LISTEN_PORT))
        addr = get_first_datagram_port(sock)
        destIP = addr[0]
        LISTEN_PORT = addr[1]
    elif args.protocol == 'TCP':
        sockStart = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sockStart.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sockStart.bind(('', LISTEN_PORT))
        sockStart.listen(1)
        sock, addr = sockStart.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print('connected:', addr)
    else:
        exit('wrong transport protocol, see help')

    return sock, LISTEN_PORT

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="side, either 'remote' or 'local' relatively to the estimated device",required=True)
    parser.add_argument("-d", "--destination", help="destination IP", required=True) 
    #parser.add_argument("-s", "--source", help="source IP, must be real", default="localhost")#, required=True)  
    parser.add_argument("-l", help="packet generation duration in sec", default='60')
    parser.add_argument("-p", "--protocol", help="transport protocol, either TCP or UDP", default="UDP")
    parser.add_argument("-o", "--object", help="file with estimated parameters, usually stored in obj/", required=True)

    parser.add_argument('-i', help="interface to dump traffic from")

    args = parser.parse_args()


    LISTEN_PORT = 8000

    #load list [estimatedParameterEM, trafficExtremeValues] from the object file 
    estimationsFileName = args.object.split('/')[len(args.object.split('/'))-1].split('.')[0]
    trafficParameters = load_obj(estimationsFileName)

    if args.m == 'remote':
        sync_remote(args.destination)
        #server side
        sock, LISTEN_PORT = init_remote_socket(args, LISTEN_PORT)

    elif args.m == 'local':
        #wait until the remote side is up
        wait_for_remote_sync()
        #client side
        sock = init_local_socket(args, LISTEN_PORT)

    else:
        exit('incorrect argument -m! see help')

    if args.i:
        pcapToSave = 'traffic_dumps/art_'+estimationsFileName+'.pcap'
        print('Saving pcap to: {}'.format(pcapToSave))
        subprocess.Popen(['timeout',args.l,'tcpdump', '-i', args.i, '-w', pcapToSave,'port','8000'])

    #allow to properly establish sockets
    time.sleep(0.5)
    
    generate_traffic(trafficParameters, args, LISTEN_PORT, sock)

    exit('Finished')


if __name__ == "__main__":
    main()