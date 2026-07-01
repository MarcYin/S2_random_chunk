from s2_random_chunk.chunks import (
    ChunkWindow,
    bounds_for_window,
    chunk_transform,
    pixel_size,
    select_random_windows,
)


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


def test_pixel_size_returns_absolute_resolution():
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0)

    assert pixel_size(transform) == (10.0, 10.0)


def test_chunk_transform_keeps_origin_when_resolution_unchanged():
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0)
    window = ChunkWindow(row_off=2, col_off=3, height=4, width=5)

    a, b, c, d, e, f = chunk_transform(transform, window, reference_resolution=10.0, target_resolution=10.0)

    assert (a, b, c, d, e, f) == (10.0, 0.0, 500030.0, 0.0, -10.0, 6999980.0)


def test_chunk_transform_scales_pixel_size_for_new_resolution():
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0)
    window = ChunkWindow(row_off=0, col_off=0, height=4, width=4)

    a, b, c, d, e, f = chunk_transform(transform, window, reference_resolution=10.0, target_resolution=20.0)

    assert (a, e) == (20.0, -20.0)
    assert (c, f) == (500000.0, 7000000.0)
