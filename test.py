from random import randint
lst = []


def fill_list():
    for i in range(10):
        yield randint(0,10)

lst += [i for i in fill_list()]
print(lst)