from coastlib.helper.progress_bar import to_ds, ProgressBar


def test_to_ds():
    assert to_ds(3) == '03'
    assert to_ds(16) == '16'


def test_progress_bar_defaults():
    n_iter = 97
    intermediate = 30
    pb = ProgressBar(total_iterations=n_iter)
    assert str(pb) == '   0% [0                                                 ] ' \
                      ' 0/97 [ETA: 00:00:00.00, Elapsed: 00:00:00.0, Speed: 0.00 it/s]'
    for _ in range(intermediate):
        pb.increment()
    assert str(pb)[:64] == '  31% [###############4                                  ] 30/97'
    for _ in range(n_iter - intermediate):
        pb.increment()
    assert str(pb)[:64] == ' 100% [##################################################] 97/97'


def test_progress_bar_customized():
    n_iter = 146
    intermediate = 121
    pb = ProgressBar(total_iterations=n_iter, bars=50, bar_items='0123456789#', prefix='HelloPB')
    assert str(pb) == 'HelloPB   0% [0                                                 ] ' \
                      '  0/146 [ETA: 00:00:00.00, Elapsed: 00:00:00.0, Speed: 0.00 it/s]'
    for _ in range(intermediate):
        pb.increment()
    assert str(pb)[:73] == 'HelloPB  83% [#########################################4        ] 121/146'
    for _ in range(n_iter - intermediate):
        pb.increment()
    assert str(pb)[:73] == 'HelloPB 100% [##################################################] 146/146'


def test_progress_bar_iprop():
    n_iter = 97
    intermediate = 30
    pb = ProgressBar(total_iterations=n_iter)
    assert str(pb) == '   0% [0                                                 ] ' \
                      ' 0/97 [ETA: 00:00:00.00, Elapsed: 00:00:00.0, Speed: 0.00 it/s]'
    for _ in range(intermediate):
        pb.i += 1
    assert str(pb)[:64] == '  31% [###############4                                  ] 30/97'
    for _ in range(n_iter - intermediate):
        pb.i += 1
    assert str(pb)[:64] == ' 100% [##################################################] 97/97'
