from collections import * 

history_length = 16
point = deque(maxlen=history_length)

for i in range(16):
    point.append(12)
print(point)

for i in range(5):
    point.append(13)
print(point)

for i in range(7):
    point.append(14)
print(point)