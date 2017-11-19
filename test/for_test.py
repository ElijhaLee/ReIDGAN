class Iter:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration()
        self.index += 1
        return self.data[self.index - 1]

    next = __next__


class T:
    def __init__(self):
        self.data = ['a', 'b', 'c']
        self.iter = Iter(self.data)

    def __iter__(self):
        return self.iter


if __name__ == '__main__':
    t = T()
    # for tt in t:
    #     print(tt)
    iter = iter(t)
    try:
        iter.next()
        iter.next()
        iter.next()
        iter.next()
    except StopIteration:
        print('ENd!')
