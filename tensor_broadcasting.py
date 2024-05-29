import torch

def broadcast_tensors(tensor1, tensor2):
    broadcasted_tensor1, broadcasted_tensor2 = torch.broadcast_tensors(tensor1, tensor2)
    print(broadcasted_tensor2.shape, broadcasted_tensor2)
    return broadcasted_tensor2.shape, broadcasted_tensor2

if __name__ == '__main__':

    tensor1 = torch.tensor([[1,2,3], [4,5,6]])
    tensor2 = torch.tensor([1,1,1])

    broadcast_tensors(tensor1, tensor2)