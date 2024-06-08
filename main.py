import numpy as np

def broadcast_tensors(tensor1, tensor2):
    broadcasted_tensor2 = np.broadcast_to(tensor2, tensor1.shape)
    print(broadcasted_tensor2.shape, broadcasted_tensor2)
    return broadcasted_tensor2.shape, broadcasted_tensor2

if __name__ == "__main__":
    tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
    tensor2 = np.array([1, 1, 1])

    broadcast_tensors(tensor1, tensor2)