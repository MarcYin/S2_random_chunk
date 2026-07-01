from s2_random_chunk.chunks import ChunkWindow, bounds_for_window, select_random_windows


def test_select_random_windows_is_seeded_and_filters_partial_edges():
    windows = (
        ChunkWindow(0, 0, 512, 512),
        ChunkWindow(0, 512, 512, 512),
        ChunkWindow(512, 0, 100, 512),
        ChunkWindow(512, 512, 512, 100),
    )

    selected = select_random_windows(windows, n_chunks=2, seed=10, full_shape=(512, 512))

    assert selected == (
        ChunkWindow(0, 0, 512, 512),
        ChunkWindow(0, 512, 512, 512),
    )


def test_bounds_for_north_up_window():
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0)
    window = ChunkWindow(row_off=2, col_off=3, height=4, width=5)

    assert bounds_for_window(transform, window) == (
        500030.0,
        6999940.0,
        500080.0,
        6999980.0,
    )
