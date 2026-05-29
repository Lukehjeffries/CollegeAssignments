import numpy as np

x = np.array([[1.0],  
              [0.5],  
              [0.2]])  


W1 = np.array([
    [0.2, -0.1, 0.4],
    [0.7,  0.5, -0.3],
    [-0.5, 0.2, 0.1],
    [0.1,  0.6, 0.3]
])

b1 = np.array([[0.1], [0.2], [0.3], [0.4]])  


z1 = np.dot(W1, x) + b1      
h = np.maximum(0, z1)         

print("Hidden layer pre-activation (z1):")
print(z1)
print("\nHidden layer output (h):")
print(h)


W2 = np.array([
    [0.3, -0.2, 0.5, 0.1],
    [-0.4, 0.6, 0.2, 0.3]
])

b2 = np.array([[0.2], [0.1]])  


z2 = np.dot(W2, h) + b2        
y = np.maximum(0, z2)          

print("\nOutput layer pre-activation (z2):")
print(z2)
print("\nFinal output (y):")
print(y)

# parameters
params = (4 * (3 + 1)) + (2 * (4 + 1))
print(f"\nTotal Parameters: {params}")
