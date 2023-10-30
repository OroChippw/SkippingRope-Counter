# -*- coding: utf-8 -*-

import datetime

class BufferList():
    def __init__(self , buffer_time , default_time=0) -> None:
        self.buffer = [default_time for _ in range(buffer_time)]
    
    def push(self , value):
        self.buffer.pop(0)
        self.buffer.append(value)
    
    def max(self):
        return max(self.buffer)
    
    def min(self):
        buffer = [value for value in self.buffer if value]
        if buffer:
            return min(buffer)
        return 0
    

class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        now = datetime.datetime.now()
        return (now - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()