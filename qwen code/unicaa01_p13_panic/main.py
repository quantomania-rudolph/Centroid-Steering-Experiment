def matmul(a, b):
    if len(a[0]) != len(b):
        raise ValueError("Incompatible dimensions for matrix multiplication")
    
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    
    return result

def demo():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    result = matmul(a, b)
    assert result == [[19, 22], [43, 50]], f"Expected [[19, 22], [43, 50]], but got {result}"
    return result
