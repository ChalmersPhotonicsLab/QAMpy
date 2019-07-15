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

import zlib
import pickle


def save_signal(fn, signal, lvl=4):
    """
    Save a signal object using zlib compression

    Parameters
    ----------
    fn : basestring
        filename
    signal : SignalObject
        the signal to save
    lvl : int, optional
        the compression to use for zlib
    """
    with  open(fn, "wb") as fp:
        sc = zlib.compress(pickle.dumps(signal, protocol=pickle.HIGHEST_PROTOCOL), level=lvl)
        fp.write(sc)

def load_signal(fn):
    """
    Load a signal object from a zlib compressed pickle file.

    Parameters
    ----------
    fn : basestring
        filename of the file

    Returns
    -------
    sig : SignalObject
        The loaded signal object

    """
    with open(fn, "rb") as fp:
        s = zlib.decompress(fp.read())
        obj = pickle.loads(s)
    return obj
