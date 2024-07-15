from decimal import Decimal
import numpy as np

x = [Decimal('0.3'), Decimal('0.01'), Decimal('1e-10'), Decimal('1e-40')]
print(x)
A = np.array(x)
A.shape = (2,2)
print(A)
print(A@A)