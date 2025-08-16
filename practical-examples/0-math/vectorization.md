## Vectorization

Vectorization is the process of converting an algorithm from operating on a single value at a time to operating on a set of values (vector) at one time.

Hence, we can use these techniques to perform operations on Numpy arrays without using loops. It only uses the pre-defined inbuilt functions like the sum function etc., and mathematical operations like "+", "-", "/" etc. for performing operations in Numpy arrays.

Vectorization also uses the concept of broadcasting for performing operations in arrays of different sizes.

<img width="1000" height="594" alt="image" src="https://github.com/user-attachments/assets/8e34673a-1bbf-4bc8-ad87-ca6b4c23d4e3" />

#### Numpy Vectorization with the np.vectorize() Function
The NumPy vectorize function (np.vectorize) is provided by the Python library. It accepts a nested sequence of objects or a NumPy array as input and returns a single NumPy array or a tuple of NumPy arrays as output. With the addition of the NumPy broadcasting rules, the np vectorize function evaluates pyfunc(user-defined function parameter) over successive tuples of the input arrays, similar to the map function in Python.

#### np.vectorize() vs. Python for a Loop – Vectorization Speed Comparison
Vectorization takes less memory and is highly optimized with the NumPy program, and can be executed faster rather than using loops. Loops iterate over an array of elements one by one, which takes lots of time, but in the case of vectorization, we can process multiple elements of the array simultaneously, which increases the speed of the program. 

**Code:**
```python
import numpy as np
import time
arr=np.arange(1,20,4)
sum=0
#program starts
st_time=time.time()
for i in arr:
  sum=sum+i
#end time
ed_time=time.time()
print("Sum of elements of array arr: ",sum)
print('Execution time:',ed_time-st_time, 'seconds')
print('Execution time:',1000*(ed_time-st_time), 'milliseconds')
```
**Output:**
```sh
The sum of elements of array arr: 45
Execution time: 6.67572021484375e-06 seconds
Execution time: 0.00667572021484375 milliseconds
```

**References:**
- https://learningactors.com/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors/
- https://dagster.io/glossary/data-vectorization
