# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn


class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.modelTeacher = modelTeacher
        self.modelStudent = modelStudent


class EnsembleTSModel_DA(nn.Module):
    def __init__(self, modelTeacher_static, modelTeacher_dynamic, modelStudent):
        super(EnsembleTSModel_DA, self).__init__()

        if isinstance(modelTeacher_static, (DistributedDataParallel, DataParallel)):
            modelTeacher_static = modelTeacher_static.module
        if isinstance(modelTeacher_dynamic, (DistributedDataParallel, DataParallel)):
            modelTeacher_dynamic = modelTeacher_dynamic.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.modelTeacher_static = modelTeacher_static
        self.modelTeacher_dynamic = modelTeacher_dynamic
        self.modelStudent = modelStudent
