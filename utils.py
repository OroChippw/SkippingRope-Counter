# -*- coding: utf-8 -*-

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