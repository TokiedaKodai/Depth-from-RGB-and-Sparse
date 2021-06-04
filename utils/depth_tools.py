import numpy as np

def unpack_bmp_bgra_to_float(bmp):
    b = bmp[:, :, 0].astype(np.int32)
    g = bmp[:, :, 1].astype(np.int32) << 16
    r = bmp[:, :, 2].astype(np.int32) << 8
    a = bmp[:, :, 3].astype(np.int32)
    depth = np.ldexp(1.0, b -
                     (128 + 24)) * (g + r + a + 0.5).astype(np.float32)
    return depth

def pack_float_to_bmp_bgra(depth):
    m, e = np.frexp(depth)
    m = (m * (256**3)).astype(np.uint64)
    bmp = np.zeros((*depth.shape[:2], 4), np.uint8)
    bmp[:, :, 0] = (e + 128).astype(np.uint8)
    bmp[:, :, 1] = np.right_shift(np.bitwise_and(m, 0x00ff0000), 16)
    bmp[:, :, 2] = np.right_shift(np.bitwise_and(m, 0x0000ff00), 8)
    bmp[:, :, 3] = np.bitwise_and(m, 0x000000ff)
    return bmp