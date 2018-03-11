# -*- coding: utf-8 -*-
#  This file is part of QAMpy.
#
#  QAMpy is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Foobar is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with QAMpy.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2018 Jochen Schröder, Mikael Mazur

import zmq
import numpy as np
import msgpack
import msgpack_numpy as msgp_npy
from qampy.core.phaserecovery import blindphasesearch

# careful we cannot use the unpatched msgpack because it messes up dictionary keys to bytes
msgp_npy.patch()


def pack_array(A):
    return msgpack.packb(A)

def unpack_array(A):
    return msgpack.unpackb(A)

def send_array(socket, A, flags=0, copy=True, track=False):
    socket.send(pack_array(A), flags=flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    return unpack_array(socket.recv(flags=flags, copy=copy, track=track))

class PPWorker(object):
    def __init__(self, send_url, receive_url, context=None):
        self.context = context or zmq.Context.instance()
        self.send_socket = self.context.socket(zmq.PUSH)
        self.send_socket.connect(send_url)
        self.receive_socket = self.context.socket(zmq.PULL)
        self.receive_socket.connect(receive_url)

    def send_msg(self, msg, success=b"OK", flags=None):
        self.send_socket.send_multipart([success, pack_array(A)], flags=flags)

    def recv_msg(self, flags=None):
        header, msg = self.collect_socket.recv_multipart(flags=flags)
        return header, unpack_array(msg)

    def process(self, flags=None):
        header, msg = self.recv_msg(flags=flags)
        try:
            result = getattr(self, header)(**msg)
            success = b"OK"
        except Exception as err:
            success = b"ERR"
            result = err.encode("ascii")
        self.send_msg(result, success)

    def run(self):
        while True:
            self.process()
        self.socket.close()

class RepWorker(object):
    def __init__(self, url, context=None):
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.connect(url)
        print("started on %s"%url)

    def send_msg(self, msg, success=b"OK", flags=0):
        self.socket.send_multipart([success, pack_array(msg)], flags=flags)

    def recv_msg(self, flags=0):
        header, msg = self.socket.recv_multipart(flags=flags)
        return header, unpack_array(msg)

    def process(self):
        header, msg = self.recv_msg()
        try:
            result = getattr(self, header.decode("ascii"))(msg)
            success = b"OK"
        except Exception as err:
            success = b"ERR"
            result = repr(err).encode("ascii")
        finally:
            self.send_msg(result, success)

    def run(self):
        while True:
            self.process()
            #self.recv_msg()
        #self.socket.close()


class DataDealer(object):
    def __init__(self, url, port=None, context=None):
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.DEALER)
        if port == None:
            self.port = self.socket.bind_to_random_port(url, min_port=5000, max_port=5100)
        else:
            self.socket.bind(u"{}:{}".format(url, port))
            self.port = port

    def send_msg(self, header, msg, identity=None, flags=0):
        msg = pack_array(msg)
        if identity is None:
            self.socket.send_multipart([b"", header, msg], flags=flags)
            #self.socket.send_multipart([b"", header])
        else:
            self.socket.send_multipart([identity, b"", header, msg], flags=flags)

    def recv_msg(self, flags=0):
        msg = self.socket.recv_multipart(flags=flags)
        if msg[0] == b"":
            if msg[1] == b"OK":
                return unpack_array(msg[2])
            else:
                raise Exception(msg)
        else:
            pass

class ResultSink(object):
    def __init__(self, url, port=None, context=None):
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.PULL)
        if port is None:
            self.socket.bind_to_random_port(url)
        else:
            self.socket.bind(url+":"+port)


class PhRecWorker(RepWorker):
    def do_phase_rec(self, pdict):
        id = pdict['id']
        E = pdict['data']
        M = pdict['Mtestangles']
        syms = pdict['symbols']
        N = pdict['N']
        Eout, ph = blindphasesearch(E, M, syms, N)
        return {'Eout':Eout, 'ph':ph, 'id': id}



