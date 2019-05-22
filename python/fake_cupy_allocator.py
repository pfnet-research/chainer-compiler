try:
    import cupy
except Exception:
    cupy = None


class use_fake_cupy_allocator:

    def __init__(self, xp):
        self.xp = xp

    def __enter__(self):
        if self.xp != cupy:
            return

        self.buf_size = 8 * 10 ** 9  # FIXME(mkusumoto): do not use fixed size
        cupy.cuda.set_allocator()
        self.buf = cupy.cuda.alloc(self.buf_size)
        self.offset = 0

        cupy.cuda.set_allocator(self._alloc)

    def _alloc(self, n):
        ret = self.buf + self.offset
        n = (n + 15) // 16 * 16
        if self.offset + n > self.buf_size:
            self.offset = (self.offset + n) % self.buf_size
        ret = self.buf + self.offset
        self.offset = (self.offset + n) % self.buf_size
        return ret

    def __exit__(self, type, value, traceback):
        if self.xp != cupy:
            return

        del self.buf

        memory_pool = cupy.get_default_memory_pool()
        cupy.cuda.set_allocator(memory_pool.malloc)
