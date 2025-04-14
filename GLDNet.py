import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation (SE) 模块：自适应通道权重
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()    # 压缩：全局平均池化全连接降维
        self.fc1 = nn.Linear(channels, channels // reduction)   # 激励：全连接升维输出通道匹配权重
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        N, C, H, W = x.size()        # 全局平均池化每个通道缩减
        w = F.adaptive_avg_pool2d(x, 1).view(N, C)      # shape: [N, C]
        w = F.relu(self.fc1(w))                        # 压缩后的通道特征
        w = torch.sigmoid(self.fc2(w))           # 输出0-1之间通道权重
        w = w.view(N, C, 1, 1)                         # 调整shape通道乘法
        return x * w                                   # 原特征按道加权调整

# 残差基本模块：包含两个卷积层，可嵌入SE注意力
class BasicBlock(nn.Module):
    expansion = 1  # 基本块不改变通道
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        # SE注意力模块
        self.se = SEBlock(out_channels) if use_se else None
        # Shortcut分支：如果尺寸或通道不匹配则使用1x1卷积调整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # 尺寸相同，捷径就是恒等映射

    def forward(self, x):
        # 主分支卷积流水线
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 插入SE模块进行通道重标定（如用）
        if self.use_se and self.se is not None:
            out = self.se(out)
        # 与捷径分支相加实现残差连接
        out += self.shortcut(x)
        # 输出再经过ReLU激活
        out = F.relu(out)
        return out

# 非局部注意力模块：实现全局动态交互
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(NonLocalBlock, self).__init__()
        inter_channels = max(1, in_channels // reduction)
        # θ, φ, g变换：将输入特征映射到低维空间以计算关系
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.phi   = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.g     = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        # 输出变换：将聚合后的特征映射回原通道维度
        self.out_conv = nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        N, C, H, W = x.size()
        theta_x = self.theta(x).view(N, -1, H * W)               # C_int = in_channels//2
        phi_x   = self.phi(x).view(N, -1, H * W)
        theta_xT = theta_x.permute(0, 2, 1).contiguous()         # [N, HW, C_int]
        f = torch.matmul(theta_xT, phi_x)                       # [N, HW, HW]
        f_weights = F.softmax(f, dim=-1)                        # 注意力权重 [N, HW, HW]
        g_x = self.g(x).view(N, -1, H * W)
        g_xT = g_x.permute(0, 2, 1).contiguous()                # [N, HW, C_int]
        y = torch.matmul(f_weights, g_xT)                       # [N, HW, C_int]
        y = y.permute(0, 2, 1).contiguous().view(N, -1, H, W)   # 恢复空间维度
        # 映射回原通道数并与输入残差相加
        out = self.out_conv(y)
        return x + out  # 输出加入残差连接



# 全局-局部动态交互网络（带SE和Non-local模块）
class GlobalLocalDynamicNet(nn.Module):
    def __init__(self, num_classes=10, base_channels=24, use_se=True, use_nonlocal=True):
        super(GlobalLocalDynamicNet, self).__init__()
        # 初始卷积层，将输入通道3提升到 base_channels
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        # 定义每个阶段的输出通道数（逐渐加倍），以及残差块数量
        ch1, ch2, ch3 = base_channels, base_channels*2, base_channels*4
        n1 = n2 = n3 = 3  # 每个阶段堆叠3个残差块
        # 第1阶段：输出通道 ch1，空间大小不变
        self.layer1 = self._make_layer(BasicBlock, in_channels=ch1, out_channels=ch1, blocks=n1, stride=1, use_se=use_se)
        # 第2阶段：输出通道 ch2，空间下采样一半
        self.layer2 = self._make_layer(BasicBlock, in_channels=ch1, out_channels=ch2, blocks=n2, stride=2, use_se=use_se)
        # 第3阶段：输出通道 ch3，空间下采样一半
        self.layer3 = self._make_layer(BasicBlock, in_channels=ch2, out_channels=ch3, blocks=n3, stride=2, use_se=use_se)
        # 非局部注意力模块（可选）
        self.nl = NonLocalBlock(ch3) if use_nonlocal else None
        # 分类器：全局池化 + 全连接输出类别
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(ch3, num_classes)
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride, use_se):
        """堆叠指定数量的残差块"""
        layers = []
        # 第一个块可能需要下采样
        layers.append(block(in_channels, out_channels, stride=stride, use_se=use_se))
        # 剩余块stride=1
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1, use_se=use_se))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积和BN
        out = F.relu(self.bn1(self.conv1(x)))
        # 三个阶段的残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 非局部全局注意力
        if self.nl is not None:
            out = self.nl(out)
        # 全局平均池化和全连接分类输出
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# modeling (参考)
#model = GlobalLocalDynamicNet(num_classes=10, base_channels=24, use_se=True, use_nonlocal=True).to(device)
