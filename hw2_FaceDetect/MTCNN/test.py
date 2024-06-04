import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from MTCNN import PNet, RNet, ONet, nms, convert_to_square
from torchvision import transforms
import time
import os

save_out="D:/test_code/MTCNN/results/test/"

class Detector:
    def __init__(self, pnet_param="./param/p_net.pth", rnet_param="./param/r_net.pth", onet_param="./param/o_net.pth",
                 isCuda=False):
        # def __init__(self, pnet_param=r"C:\Users\Administrator\Desktop\Learnn\DL\MTCNN\60k\p_net.pth", rnet_param=r"C:\Users\Administrator\Desktop\Learnn\DL\MTCNN\60k\r_net.pth",
        #              onet_param=r"C:\Users\Administrator\Desktop\Learnn\DL\MTCNN\60k\o_net.pth",
        #              isCuda=False):
        self.isCuda = isCuda

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param, map_location='cpu'))
        self.rnet.load_state_dict(torch.load(rnet_param, map_location='cpu'))
        self.onet.load_state_dict(torch.load(onet_param, map_location='cpu'))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        # 记住一定要使用model.eval()来固定dropout和归一化层，否则每次推理会生成不同的结果

        self.__image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect(self, image):
        start_time = time.time()
        # print("===================")
        pnet_boxes = self.__pnet_detect(image)
        # print("***********************")
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)  # p网络输出的框和原图像输送到R网络中，O网络将框扩为正方形再进行裁剪，再缩放
        # print( rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()

        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes
        # return rnet_boxes

    def __pnet_detect(self, img):

        boxes = []
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len >= 12:
            img_data = self.__image_transform(img)
            # img_data = img_data.unsqueeze_(0)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)  # 升维度（新版pytorch可以删掉）

            _cls, _offest = self.pnet(img_data)  # NCHW
            # print(_cls.shape)    #torch.Size([1, 1, 1290, 1938])
            # print(_offest.shape)    #torch.Size([1, 4, 1290, 1938])

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            # _cls[0][0].cpu().data去掉NC，  _offest[0]去掉N
            # print(_cls.shape)       #torch.Size([1, 1, 1290, 1938])
            # print(_offest.shape)     #torch.Size([1, 4, 1290, 1938])

            idxs = torch.nonzero(torch.gt(cls, 0.6))  # 取出置信度大于0.6的索引
            # print(idxs.shape)   #N2     #torch.Size([4639, 2])

            for idx in idxs:  # idx里面就是一个h和一个w
                # print(idx)    #tensor([ 102, 1904])
                # print(offest)
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))  # 反算
            scale *= 0.709
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            # print(min_side_len)
            min_side_len = np.minimum(_w, _h)
        return nms(np.array(boxes), 0.3)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):  # side_len=12建议框大大小

        _x1 = int(start_index[1] * stride) / scale  # 宽，W，x
        _y1 = int(start_index[0] * stride) / scale  # 高，H,y
        _x2 = int(start_index[1] * stride + side_len) / scale
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1  # 偏移量
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]  # 通道层面全都要[C, H, W]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        boxes = []


        idxs, _ = np.where(_cls > 0.6)
        for idx in idxs:  # 只是取出合格的
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])
            # print(len(utils.nms(np.array(boxes), 0.3)))
        print("""""""""""""""""""""""""""""""""""""""""")

        return nms(np.array(boxes), 0.3)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(_cls > 0.97)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])

        return nms(np.array(boxes), 0.3, isMin=True)


if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        path = r"D:\test_code\MTCNN\celeba\images\test"  # 遍历文件夹内的图片
        for name in os.listdir(path):
            print(name)
            img = os.path.join(path, name)
            image_file = img
            # image_file = r"1.jpg"
            # print(image_file)
            detector = Detector()

            with Image.open(image_file) as im:
                boxes = detector.detect(im)
                # print(im,"==========================")
                # print(boxes.shape)
                imDraw = ImageDraw.Draw(im)
                for box in boxes:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    # print(x1)
                    # print(y1)
                    # print(x2)
                    # print(y2)

                    # print(box[4])
                    cls = box[4]
                    imDraw.rectangle((x1, y1, x2, y2), outline='red')
                    #font = ImageFont.truetype(r"C:\Windows\Fonts\simhei", size=20)
                    # imDraw.text((x1, y1), "{:.3f}".format(cls), fill="red", font=font)
                y = time.time()
                print(y - x)
                #im.show()
                out_name=name.split('.')[0]
                save_path=save_out + out_name + '.jpg'
                im.save(save_path)
