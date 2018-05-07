from ru_net import Ru_net

class Model(object):
    def __init__(self):
        self.nets = {}

    def get_net(self, nx, ny):
        netname = str(nx) + ',' + str(ny)
        if netname in self.nets:
            return self.nets[netname]
        else:
            newnet = Ru_net(nx, ny)
            self.nets[netname] = newnet
            return newnet

    def train(self, nx, ny):
        net = self.get_net(nx, ny)

if __name__ == '__main__':
    model = Model()
