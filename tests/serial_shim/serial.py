"""Minimal stand-in for pyserial on POSIX ttys, used only by tests/test_hil_host.py when
pyserial is not installed. Implements just what tools/hil.py uses."""
import os
import select
import termios
import time


class Serial:
    def __init__(self, port=None, baudrate=9600, timeout=None):
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self.dtr = self.rts = True
        self.fd = None
        if port:
            self.open()

    def open(self):
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0                      # raw: no input/output/local processing
        a[2] |= termios.CS8 | termios.CREAD | termios.CLOCAL
        termios.tcsetattr(self.fd, termios.TCSANOW, a)

    def write(self, data):
        view = memoryview(data)
        while view:
            n = os.write(self.fd, view)
            view = view[n:]
        return len(data)

    def read(self, n):
        out = bytearray()
        end = None if self.timeout is None else time.time() + self.timeout
        while len(out) < n:
            wait = None if end is None else max(0.0, end - time.time())
            r, _, _ = select.select([self.fd], [], [], wait)
            if not r:
                break
            out += os.read(self.fd, n - len(out))
        return bytes(out)

    def readline(self):
        out = bytearray()
        while not out.endswith(b"\n"):
            c = self.read(1)
            if not c:
                break
            out += c
        return bytes(out)

    def reset_input_buffer(self):
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def close(self):
        os.close(self.fd)
