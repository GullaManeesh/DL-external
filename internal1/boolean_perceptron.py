import numpy as np

def step(z):
  return 1 if z>=0 else 0

def perceptron(x,w,b):
  z=np.dot(x,w)+b
  return step(z)

input = np.array([[0,0],[0,1],[1,0],[1,1]])

#and
for x in input:
  print(f"{x}:{perceptron(x,[1,1],-1.5)}")


#or
for x in input:
  print(f"{x}:{perceptron(x,[1,1],-0.5)}")


#nor
for x in input:
  print(f"{x}:{perceptron(x,[-1,-1],0.5)}")


#nand
for x in input:
  print(f"{x}:{perceptron(x,[-1,-1],1.5)}")


# NOT
for x in [[0],[1]]:
    print(f"{x}:{perceptron(x,[-1],0.5)}")
