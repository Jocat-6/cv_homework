import os
import numpy as np
from PIL import Image
import torch
from h5py import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report, recall_score
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

#from MTCNN.tool import utils

save_path = r"D:\test_code\MTCNN\dataSet"  # 生成样本的总的保存路径
float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]  # 控制正负样本比例，（控制比例？）
celeba_path = r"D:\test_code\MTCNN\celeba"

def IOU(box, boxes, isMin = False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h                                                 #重合部分的面积
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))   #真正除法的运算结果     #大框套小框
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))    #正常的交并比

    return ovr                                                     #返回结果

def nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:  # 防止程序出错
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]  # 返回的是数组值从小到大的索引值
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        # print(iou(a_box, b_boxes))

        index = np.where(IOU(a_box, b_boxes, isMin) < thresh)  # 返回满足条件的索引
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)


def convert_to_square(bbox):
    # print(bbox)
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox

def gen_sample(face_size):
    print("gen size:{} image".format(face_size))
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")  # 仅仅生成路径名
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:  # 生成路径
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    positive_anno_file = open(positive_anno_filename, "w")
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")

    train_txt=os.path.join(celeba_path,"train.txt")

    for i, line in enumerate(open(train_txt)):
        strs = line.split()
        #path,cls,x1,y1,x2,y2,lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
        image_file = os.path.join(celeba_path,strs[0])
        print(i,',',image_file)
        with Image.open(image_file) as img:
            img_w, img_h = img.size  # 原图
            x1 = float(strs[2].strip())
            y1 = float(strs[3].strip())
            w = float(strs[4].strip())  # 人脸框
            h = float(strs[5].strip())
            x2 = float(x1 + w)
            y2 = float(y1 + h)

            px1 = 0  # float(strs[5].strip())
            py1 = 0  # float(strs[6].strip())
            px2 = 0  # float(strs[7].strip())
            py2 = 0  # float(strs[8].strip())
            px3 = 0  # float(strs[9].strip())
            py3 = 0  # float(strs[10].strip())
            px4 = 0  # float(strs[11].strip())
            py4 = 0  # float(strs[12].strip())
            px5 = 0  # float(strs[13].strip())
            py5 = 0  # float(strs[14].strip())

            if x1 < 0 or y1 < 0 or w < 0 or h < 0:  # 跳过坐标值为负数的
                continue

            boxes = [[x1, y1, x2, y2]]  # 当前真实框四个坐标（根据中心点偏移）， 二维数组便于IOU计算

            # 求中心点坐标
            cx = x1 + w / 2
            cy = y1 + h / 2
            side_len = max(w, h)
            seed = float_num[np.random.randint(0,
                                               len(float_num))]  # 取0到9之间的随机数作为索引     #len(float_num) = 9 #float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
            count = 0
            for _ in range(4):
                _side_len = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))  # 生成框
                _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))  # 中心点作偏移
                _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))

                _x1 = _cx - _side_len / 2  # 左上角
                _y1 = _cy - _side_len / 2
                _x2 = _x1 + _side_len  # 右下角
                _y2 = _y1 + _side_len

                if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:  # 左上角的点是否偏移到了框外边，右下角的点大于图像的宽和高
                    continue

                offset_x1 = (x1 - _x1) / _side_len  # 得到四个偏移量
                offset_y1 = (y1 - _y1) / _side_len
                offset_x2 = (x2 - _x2) / _side_len
                offset_y2 = (y2 - _y2) / _side_len

                offset_px1 = 0  # (px1 - x1_) / side_len     #offset偏移量
                offset_py1 = 0  # (py1 - y1_) / side_len
                offset_px2 = 0  # (px2 - x1_) / side_len
                offset_py2 = 0  # (py2 - y1_) / side_len
                offset_px3 = 0  # (px3 - x1_) / side_len
                offset_py3 = 0  # (py3 - y1_) / side_len
                offset_px4 = 0  # (px4 - x1_) / side_len
                offset_py4 = 0  # (py4 - y1_) / side_len
                offset_px5 = 0  # (px5 - x1_) / side_len
                offset_py5 = 0  # (py5 - y1_) / side_len

                crop_box = [_x1, _y1, _x2, _y2]
                face_crop = img.crop(crop_box)  # 图片裁剪
                face_resize = face_crop.resize((face_size, face_size))  # 对裁剪后的图片缩放

                iou = IOU(crop_box, np.array(boxes))[0]
                if iou > 0.65:  # 可以自己修改
                    positive_anno_file.write(
                        "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            positive_count, 1, offset_x1, offset_y1,
                            offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                            offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    positive_anno_file.flush()  # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区
                    face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                    # print("positive_count",positive_count)
                    positive_count += 1
                elif 0.65 > iou > 0.4:
                    part_anno_file.write(
                        "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            part_count, 2, offset_x1, offset_y1, offset_x2,
                            offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                            offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    part_anno_file.flush()
                    face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    # print("part_count", part_count)
                    part_count += 1
                elif iou < 0.1:
                    negative_anno_file.write(
                        "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                    negative_anno_file.flush()
                    face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    # print("negative_count", negative_count)
                    negative_count += 1

                count = positive_count + part_count + negative_count

    positive_anno_file.close()
    negative_anno_file.close()
    part_anno_file.close()


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 10*10*10
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # 5*5*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 3*3*16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 1*1*32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=1, stride=1,
                               padding=0, dilation=1, groups=1)

    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        y = self.conv3(y)
        # y = torch.reshape(y, [y.size(0), -1])
        y = self.conv4(y)
        # print(y)
        # print()
        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        # print(category.shape)
        # print(offset.shape)
        # print("--------------------")
        return category, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 22*22*28
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 1)  # 11*11*28
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 9*9*48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 0)  # 4*4*48
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1,
                      padding=0, dilation=1, groups=1),  # 3*3*64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = torch.reshape(y, [y.size(0), -1])
        # print(y.shape)
        y = self.fc1(y)
        # print(y.shape)
        y = self.fc2(y)
        # print(y.shape)

        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        return category, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 46*46*32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 1)  # 23*23*32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 21*21*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2)  # 10*10*64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1),  # 8*8*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # 4*4*64
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1,
                      padding=0, dilation=1, groups=1),  # 3*3*128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = self.conv4(y)
        # print(y.shape,"==========")
        y = torch.reshape(y, [y.size(0), -1])
        # print(y.shape)

        y = self.fc1(y)
        # print(y.shape)
        y = self.fc2(y)
        # print(y.shape)
        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        return category, offset


class FaceDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        self.dataset = []
        # f1 = os.listdir(os.path.join(data_path, "negative"))
        # f2 = os.listdir(os.path.join(data_path, "positive"))
        # f3 = os.listdir(os.path.join(data_path, "part"))
        l1 = open(os.path.join(data_path, "negative.txt")).readlines()
        for l1_filename in l1:
            self.dataset.append([os.path.join(data_path, l1_filename.split(" ")[0]), l1_filename.split(" ")[1:6]])
            # print(self.dataset)
        # exit()
        l2 = open(os.path.join(data_path, "positive.txt")).readlines()
        for l2_filename in l2:
            self.dataset.append([os.path.join(data_path, l2_filename.split(" ")[0]), l2_filename.split(" ")[1:6]])
        l3 = open(os.path.join(data_path, "part.txt")).readlines()
        for l3_filename in l3:
            self.dataset.append([os.path.join(data_path, l3_filename.split(" ")[0]), l3_filename.split(" ")[1:6]])
        # print(self.dataset.shape())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        # print(data[0])
        img_tensor = self.trans(Image.open(data[0]))
        category = torch.tensor(float(data[1][0])).reshape(-1)
        # print(category.shape,"9999999999999999999999")
        offset = torch.tensor([float(data[1][1]), float(data[1][2]), float(data[1][3]), float(data[1][4])])

        return img_tensor, category, offset

    def trans(self, x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ])
        ])(x)


class Trainer:
    def __init__(self, net, save_path, dataset_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # 判断是否有gpu
        else:
            self.device = torch.device("cpu")
        self.net = net.to(self.device)  # 通用的属性加self
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.cls_loss_fn = nn.BCELoss()  # 置信度损失函数
        self.offset_loss_fn = nn.MSELoss()  # 坐标偏移量损失函数

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):  # 是否有已经保存的参数文件
            net.load_state_dict(torch.load(self.save_path, map_location='cpu'))
        else:
            print("NO Param")

    def trainer(self, stop_value):
        ''''''''''''''''''''
        plt_loss=[[] for i in range(19)]
        plt_cls=[[] for i in range(19)]
        plt_offset=[[] for i in range(19)]
        plt_acc=[]
        plt_recall=[]
        ''''''''''''''''''''
        faceDataset = FaceDataset(self.dataset_path)  # 实例化对象
        # dataloader = DataLoader(faceDataset, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
        dataloader = DataLoader(faceDataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        loss = 0
        self.net.train()
        while True:
            loss1 = 0
            epoch = 0
            cla_label = []
            cla_out = []
            offset_label = []
            offset_out = []
            plt.ion()
            e = []
            r = []
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                img_data_ = img_data_.to(self.device)  # 得到的三个值传入到CPU或者GPU
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)

                _output_category, _output_offset = self.net(img_data_)  # 输出置信度和偏移值

                # print(_output_category.shape)    #torch.Size([10, 1, 1, 1])
                # print(_output_offset.shape,"=================")   #torch.Size([10, 4, 1, 1])

                output_category = _output_category.view(-1, 1)  # 转化成NV结构
                output_offset = _output_offset.view(-1, 4)
                # print(output_category.shape)
                # print(output_offset.shape, "=================")

                # 正样本和负样本用来训练置信度
                category_mask = torch.lt(category_,
                                         2)  # 小于2   #一系列布尔值  逐元素比较input和other ， 即是否 \( input < other \)，第二个参数可以为一个数或与第一个参数相同形状和类型的张量。
                category = torch.masked_select(category_,
                                               category_mask)  # 取到对应位置上的标签置信度   #https://blog.csdn.net/SoftPoeter/article/details/81667810
                # torch.masked_select()根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量，

                # 上面两行等价于category_mask = category[category < 2]
                output_category = torch.masked_select(output_category, category_mask)  # 输出的置信度
                # print(output_category)
                # print(category)
                cls_loss = self.cls_loss_fn(output_category, category)  # 计算置信度的损失

                offset_mask = torch.gt(category_, 0)
                offset = torch.masked_select(offset_, offset_mask)
                output_offset = torch.masked_select(output_offset, offset_mask)
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 计算偏移值的损失

                loss = cls_loss + offset_loss
                writer = SummaryWriter()
                writer.add_scalars("loss", {"train_loss": loss}, epoch)  # 标量
                self.optimizer.zero_grad()  # 更新梯度反向传播
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()  # 将损失转达CPU上计算，此处的损失指的是每一批次的损失
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                print("epoch:", epoch, "loss:", loss, " cls_loss:", cls_loss, " offset_loss", offset_loss)
                ''''''''''''''''''''
                plt_loss[epoch].append(loss)
                plt_cls[epoch].append(cls_loss)
                plt_offset[epoch].append(offset_loss)
                ''''''''''''''''''''
                epoch += 1

                cla_out.extend(output_category.detach().cpu())
                cla_label.extend(category.detach().cpu())
                offset_out.extend(output_offset.detach().cpu())
                offset_label.extend(offset.detach().cpu())

                print("cla     :") #
                print("r2       :", r2_score(cla_label, cla_out))
                print("offset     :")
                print("r2       :", r2_score(offset_label, offset_out))
                print("total    :")
                print("r2       :", r2_score(offset_label + cla_label, offset_out + cla_out))

                #############################################
                e.append(i)                                                                            #画出r2
                r.append(r2_score(offset_label+cla_label, offset_out+cla_out))
                plt.clf()
                plt.plot(e, r)
                plt.pause(0.01)
                #
                cla_out = list(map(int, cla_out))                                                      #map方法可以将列表中的每一个元素转为相对应的元素类型
                cla_label = list(map(int, cla_label))
                offset_out = list(map(int, offset_out))
                offset_label = list(map(int, offset_label))
                #
                acc_score=accuracy_score(offset_label + cla_label, offset_out + cla_out)
                recall=recall_score(offset_label + cla_label, offset_out + cla_out)
                plt_acc.append(acc_score)
                plt_recall.append(recall)
                print("accuracy_score :", acc_score)    #求的是每一批里面的
                print("recall_score :", recall)
                print("confusion_matrix :")
                print(confusion_matrix(offset_label + cla_label, offset_out + cla_out))
                print(classification_report(offset_label + cla_label, offset_out + cla_out))
                #############################################
                cla_out = []
                cla_label.clear()
                offset_out.clear()
                offset_label.clear()
                # flops, params = thop.profile_origin(self.net, (img_data_,))  # 查看参数量
                # flops, params = thop.clever_format((flops, params), format=("%.2f"))
                # print("flops:", flops, "params:", params)
                print()


            torch.save(self.net.state_dict(), self.save_path)
            # 保存模型的推理过程的时候，只需要保存模型训练好的参数，
            # 使用torch.save()保存state_dict，能够方便模型的加载
            print("save success")

            # print(plt_loss)

            if loss < stop_value:
                break
        #画图
        x_axis_data = [i for i in range(1, 20)]
        # loss
        losses=[]
        for ls in plt_loss:
            losses.append(np.mean(ls))
        # 类别分类损失 cls_loss
        cls_losses=[]
        for ls in plt_cls:
            cls_losses.append(np.mean(ls))
        # 偏移量损失 offset_loss
        offset_losses=[]
        for ls in plt_offset:
            offset_losses.append(np.mean(ls))
        plt.plot(x_axis_data, losses, 'b*--', alpha=0.5, linewidth=1, label='loss')
        plt.plot(x_axis_data, cls_losses, 'rs--', alpha=0.5, linewidth=1, label='cls_loss')
        plt.plot(x_axis_data, offset_losses, 'go--', alpha=0.5, linewidth=1, label='offset_loss')
        plt.xlabel('epoch')
        plt.ylabel('losses_value')
        plt.show()
        plt.plot(x_axis_data,plt_acc,'b*--', alpha=0.5, linewidth=1, label='acc')
        plt.legend()  # 显示上面的label
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
        plt.plot(x_axis_data, plt_recall, 'b*--', alpha=0.5, linewidth=1, label='recall')
        plt.legend()  # 显示上面的label
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.show()





if __name__ == '__main__':
    # gen_sample(12)
    # gen_sample(24)
    # gen_sample(48)

    data_path = r"D:\Users\74178\Desktop\computer vision\MTCNN\dataSet\12"
    mydata = FaceDataset(data_path)
    data = DataLoader(mydata, 3, shuffle=True)
    for i, (x1, y1, y2) in enumerate(data):
        print(x1)
        print(x1.shape)
        print(y1)
        print(y2.shape)
        print()
        exit()