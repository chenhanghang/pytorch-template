import torch


def accuracy(output, target):
    with torch.no_grad(): # 注意屏蔽梯度
        pred = torch.argmax(output, dim=1) # 返回最大值的下标
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1] # topk[1] 下标， top[0] 值
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
