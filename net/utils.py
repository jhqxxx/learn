from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    '''warmup_training learning rate scheduler
    Args:
        optimizer:optimizer e.g SGD
        total_iters: total_iters of warmup phase
    '''
    def __init__(self, optimizer, total_iters, last_epoch=-1) -> None:
        super().__init__(optimizer, last_epoch)
        self.total_iters = total_iters
    
    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
