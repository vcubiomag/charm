import pytest

from charm.utils.parallel import BackgroundTasks


def _square(x):
    return x * x


def _fail(x):
    raise ValueError(x)


def test_results_come_back_in_order():
    assert BackgroundTasks(2, _square, [(i,) for i in range(5)]).wait() == [0, 1, 4, 9, 16]


def test_exceptions_reach_the_caller():
    with pytest.raises(ValueError):
        BackgroundTasks(2, _fail, [(1,)]).wait()
