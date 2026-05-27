
import numpy as np

def mp(x,threshold):
  z=np.sum(x)

  return 1 if z>=threshold else 0

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])


#AND

for x in inputs:
  print(f"{x}:{mp(x,2)}")
print()
#OR
for x in inputs:
  print(f"{x}:{mp(x,1)}")


