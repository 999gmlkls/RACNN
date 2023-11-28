import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset  # 这里添加了 Dataset
from torchvision import transforms
from torch.autograd import Variable
from torchvision.models import mobilenet
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
device = torch.device("cpu")
import wandb
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 初始化wandb
wandb.init(project='air', name='4444')


class CustomDataset(Dataset):
    def __init__(self, root, train=True):  # 添加 train 参数
        self.root = root
        # 根据 train 参数选择数据转换
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((448, 448)),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                #transforms.RandomCrop(224, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self._imgpath = []
        self._imglabel = []
        self.classes = set()  # 初始化 classes 集合

        # 遍历目录以收集图像路径和标签
        for label, cls_folder in enumerate(sorted(os.listdir(root))):
            cls_path = os.path.join(root, cls_folder)
            if os.path.isdir(cls_path):
                self.classes.add(cls_folder)  # 添加类名
                for img_file in sorted(os.listdir(cls_path)):
                    img_path = os.path.join(cls_path, img_file)
                    self._imgpath.append(img_path)
                    self._imglabel.append(label)

        self.classes = sorted(list(self.classes))  # 转换为列表并排序
    def get_classes(self):
        """返回数据集中的所有类别"""
        return self.classes

    def __getitem__(self, index):
        img_path = self._imgpath[index]
        img = Image.open(img_path)  # 使用PIL来读取图像

        # 应用转换
        if self.transform is not None:
            img = self.transform(img)

        cls = self._imglabel[index]
        return img, cls

    def __len__(self):
        return len(self._imgpath)


class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        def h(_x): return 1 / (1 + torch.exp(-10 * _x.float()))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.to(device), y.to(device)

        ret = []
        for i in range(images.size(0)):
            if i >= len(locs):
                break

            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tl = tl if tl > (in_size/3) else in_size/3
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size-tl else in_size-tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size-tl else in_size-tl

            w_off = int(tx-tl) if (tx-tl) > 0 else 0
            h_off = int(ty-tl) if (ty-tl) > 0 else 0
            w_end = int(tx+tl) if (tx+tl) < in_size else in_size
            h_end = int(ty+tl) if (ty+tl) < in_size else in_size

            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk

            xatt_cropped = xatt[:, w_off: w_end, h_off: h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.interpolate(before_upsample, size=(224, 224), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size/3*2)
        short_size = (in_size/3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size)+(x >= long_size)+(y < short_size)+(y >= long_size)) > 0).float()*2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret



class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)
class RACNN(nn.Module):
    def __init__(self, num_classes, img_scale=448):
        super(RACNN, self).__init__()

        self.b1 = mobilenet.mobilenet_v2(num_classes=num_classes)
        self.b2 = mobilenet.mobilenet_v2(num_classes=num_classes)
        self.b3 = mobilenet.mobilenet_v2(num_classes=num_classes)
        self.classifier1 = nn.Linear(320, num_classes)
        self.classifier2 = nn.Linear(320, num_classes)
        self.classifier3 = nn.Linear(320, num_classes)
        self.feature_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.crop_resize = AttentionCropLayer()
        self.apn1 = nn.Sequential(
            nn.Linear(320 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.apn2 = nn.Sequential(
            nn.Linear(320 * 7 * 7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.echo = None

    def forward(self, x):
        batch_size = x.shape[0]
        rescale_tl = torch.tensor([1, 1, 0.5], requires_grad=False)

        # 第一尺度处理
        feature_s1 = self.b1.features[:-1](x)  # 提取特征
        pool_s1 = self.feature_pool(feature_s1)  # 池化
        #print(feature_s1.shape)  # 打印 feature_s1 的大小

        _attention_s1 = self.apn1(feature_s1.view(-1, 320 * 14 * 14))  # 注意力预测
        attention_s1 = _attention_s1 * rescale_tl  # 调整注意力尺度
        resized_s1 = self.crop_resize(x, attention_s1 * x.shape[-1])  # 裁剪和调整尺寸

        # 第二尺度处理
        feature_s2 = self.b2.features[:-1](resized_s1)  # 提取特征
        pool_s2 = self.feature_pool(feature_s2)  # 池化
        #print(feature_s2.shape)  # 打印 feature_s1 的大小
        _attention_s2 = self.apn2(feature_s2.view(-1, 320 * 7 * 7))  # 注意力预测
        attention_s2 = _attention_s2 * rescale_tl  # 调整注意力尺度
        resized_s2 = self.crop_resize(resized_s1, attention_s2 * resized_s1.shape[-1])  # 裁剪和调整尺寸

        # 第三尺度处理
        feature_s3 = self.b3.features[:-1](resized_s2)  # 提取特征
        pool_s3 = self.feature_pool(feature_s3)  # 池化

        # 分类预测
        pred1 = self.classifier1(pool_s1.view(-1, 320))
        pred2 = self.classifier2(pool_s2.view(-1, 320))
        pred3 = self.classifier3(pool_s3.view(-1, 320))

        # 输出
        return [pred1, pred2, pred3], [feature_s1, feature_s2], [attention_s1, attention_s2], [resized_s1, resized_s2]

    def __get_weak_loc(self, features):
        ret = []  # search regions with the highest response value in conv5
        for i in range(len(features)):
            resize = 224 if i >= 1 else 448
            response_map_batch = F.interpolate(features[i], size=[resize, resize], mode="bilinear").mean(
                1)  # mean alone channels
            ret_batch = []
            for response_map in response_map_batch:
                argmax_idx = response_map.argmax()
                ty = (argmax_idx % resize)
                argmax_idx = (argmax_idx - ty) / resize
                tx = (argmax_idx % resize)
                ret_batch.append(
                    [(tx * 1.0 / resize).clamp(min=0.25, max=0.75), (ty * 1.0 / resize).clamp(min=0.25, max=0.75),
                     0.25])  # tl = 0.25, fixed
            ret.append(torch.Tensor(ret_batch))
        return ret

    def __echo_pretrain_apn(self, inputs, optimizer):
        inputs = Variable(inputs).to(device)
        _, features, attens, _ = self.forward(inputs)
        weak_loc = self.__get_weak_loc(features)
        optimizer.zero_grad()
        weak_loss1 = F.smooth_l1_loss(attens[0], weak_loc[0].cuda())
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1].cuda())
        loss = weak_loss1 + weak_loss2
        loss.backward()
        optimizer.step()
        return loss.item()


    @staticmethod
    def multitask_loss(logits, targets):
        loss = []
        for i in range(len(logits)):
            loss.append(F.cross_entropy(logits[i], targets))
        loss = torch.sum(torch.stack(loss))
        return loss

    @staticmethod
    def rank_loss(logits, targets, margin=0.05):
        preds = [F.softmax(x, dim=-1) for x in logits]
        set_pt = [[scaled_pred[batch_inner_id][target] for scaled_pred in preds] for batch_inner_id, target in enumerate(targets)]
        loss = 0
        for batch_inner_id, pts in enumerate(set_pt):
            loss += (pts[0] - pts[1] + margin).clamp(min=0)
            loss += (pts[1] - pts[2] + margin).clamp(min=0)
        return loss

    def __echo_backbone(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)
        logits, _, _, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.multitask_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def __echo_apn(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)
        logits, _, _, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.rank_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def mode(self, mode_type):
        assert mode_type in ['pretrain_apn', 'apn', 'backbone']
        if mode_type == 'pretrain_apn':
            self.echo = self.__echo_pretrain_apn
            self.eval()
        if mode_type == 'backbone':
            self.echo = self.__echo_backbone
            self.train()
        if mode_type == 'apn':
            self.echo = self.__echo_apn
            self.eval()




def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _, _, _ = model(images)
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs[0].data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum().item()

    train_accuracy = 100. * correct / total
    avg_train_loss = train_loss / len(train_loader)
    return train_accuracy, avg_train_loss


def test(model, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _, _ = model(images)
            loss = criterion(outputs[0], labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    return test_accuracy, test_loss / len(test_loader)


# 加载数据集
train_dataset = CustomDataset(root='15cm/train', train=True)
test_dataset = CustomDataset(root='15cm/test', train=False)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

# 输出训练集和测试集的类别和大小
print("Training dataset classes:", train_dataset.get_classes())
print("Training dataset size:", len(train_dataset))  # 使用 len() 而不是 get_dataset_size()

print("Testing dataset classes:", test_dataset.get_classes())
print("Testing dataset size:", len(test_dataset))  # 使用 len() 而不是 get_dataset_size()


# 初始化模型、优化器、损失函数和调度器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RACNN(num_classes=len(train_dataset.classes)).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# 训练和测试循环
train_accuracies = []
test_accuracies = []
test_losses = []

best_accuracy = 0
total_epochs = 2

for epoch in range(total_epochs):
    train_acc, avg_train_loss = train(model, train_loader, optimizer, criterion, epoch)
    test_acc, avg_test_loss = test(model, test_loader, criterion, epoch)

    # 使用 wandb 记录性能指标
    wandb.log({"epoch": epoch, "train_accuracy": train_acc, "train_loss": avg_train_loss, "test_accuracy": test_acc, "test_loss": avg_test_loss})

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    test_losses.append(avg_test_loss)
    scheduler.step()

    # 更新最佳模型（基于验证集准确率）
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        # 保存模型到本地
        torch.save(model.state_dict(), 'best_model.pth')

    print(f'Epoch {epoch}: Train Accuracy: {train_acc:.2f}%, Avg Train Loss: {avg_train_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Avg Test Loss: {avg_test_loss:.4f}')



