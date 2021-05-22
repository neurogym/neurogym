import pytest

import numpy as np
from neurogym.utils.random import TruncExp
from neurogym.wrappers import RandomGroundTruth
from neurogym.utils.scheduler import RandomSchedule, RandomBlockSchedule

def test_truncexp():
    te = TruncExp(vmean=100)
    te.seed(0)
    a = [te() for i in range(1000)]
    te.seed(0)
    b = [te() for i in range(1000)]

    assert (np.array(a) == np.array(b)).all(), 'TruncExp not reproducible'


def test_randomschedule():
    schedule = RandomSchedule(10)
    schedule.reset()
    schedule.seed(0)
    a = [schedule() for i in range(1000)]
    schedule.reset()
    schedule.seed(0)
    b = [schedule() for i in range(1000)]
    assert (np.array(a) == np.array(b)).all(), 'RandomSchedule not ' \
                                               'reproducible'

    schedule = RandomBlockSchedule(10, block_lens=[5]*10)
    schedule.reset()
    schedule.seed(0)
    a = [schedule() for i in range(1000)]
    schedule.reset()
    schedule.seed(0)
    b = [schedule() for i in range(1000)]
    assert (np.array(a) == np.array(b)).all(), 'RandomBlockSchedule not ' \
                                               'reproducible'
