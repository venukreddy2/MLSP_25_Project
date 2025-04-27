from .semseg import SemsegMeter
from .human_parts import HumanPartsMeter
from .normals import NormalsMeter
from .saliency import  SaliencyMeter
from .edge import EdgeMeter
from .depth import DepthMeter

class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, config, tasks):
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(config, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict

def get_single_task_meter(config, task):
    """ Retrieve a meter to measure the single-task performance """

    # ignore index based on transforms.AddIgnoreRegions
    if task == 'semseg':
        return SemsegMeter(ignore_idx=config.ignore_index, dataset=config.dataset)

    elif task == 'human_parts':
        return HumanPartsMeter(ignore_idx=config.ignore_index)

    elif task == 'normals':
        return NormalsMeter(ignore_index=config.ignore_index) 

    elif task == 'sal':
        return SaliencyMeter(ignore_index=config.ignore_index, threshold_step=0.05, beta_squared=0.3)

    elif task == 'edge': # just for reference
        return EdgeMeter(pos_weight=0.95, ignore_index=config.ignore_index)

    elif task == "depth":
        return DepthMeter(max_depth=80.0, min_depth=0)
    
    else:
        raise NotImplementedError
