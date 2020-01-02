

class MyObj:

    def __init__(self, name):
        self.name = name


def foo(obj):
    print (obj.name)
    obj.name = "123"
    print (obj.name)


def main():
    myObj = MyObj ("abc")

    print (myObj.name)
    foo(myObj)
    print (myObj.name)

main()