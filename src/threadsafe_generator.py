import threading


class ThreadsafeIterator(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing ciall to the `next` method of given iterator/generator.
    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        assert self.lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """
    def g(*a, **kw):
        return ThreadsafeIterator(f(*a, **kw))
    return g
