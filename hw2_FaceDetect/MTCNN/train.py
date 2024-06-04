from MTCNN import PNet,RNet,ONet
from MTCNN import Trainer
import os
if __name__ == '__main__':

    net_P = PNet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = Trainer(net_P, './param/p_net.pth', r".\dataSet\12")
    trainer.trainer(0.01)
    print('P done')

    net_R = RNet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = Trainer(net_R, './param/r_net.pth', r".\dataSet\24")
    trainer.trainer(0.001)
    print('R done')

    net_O = ONet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = Trainer(net_O, './param/o_net.pth', r".\dataSet\48")
    trainer.trainer(0.0003)
    print('O done')