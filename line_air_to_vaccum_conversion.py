import numpy as np
lineair=np.array([7318.986 , 7319.99])
s=10**4/lineair
n=1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)

linevac=lineair*n
print(linevac)