import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class PConv(nn.Module):
    # PWConv Block
    def __init__(self,
                 dim=int, 
                 n_div=int, #  Set to 4
                 forward= "split_cat",
                 kernel_size=3,
                 g=True): # g=True To call PWconv, g=False for PConv
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.g1 = int(dim / 4) if g is True else 1
        self.conv=nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, stride=1,padding=(kernel_size -1) // 2, groups = self.g1, bias=False)
        if forward == "slicing":
            self.forward = self.forward_slicing 
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    def forward_slicing(self, x):
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x
    def forward_split_cat(self, x):
        x1, x2 = torch.split(x,[self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x=torch.cat((x1,x2),1)
        return x

class FasterNetBlock(nn.Module):
    # FasterNetBlock Block
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = PConv(c1, 2, "split_cat", 3)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class FasterNeXt(nn.Module):
    # FasterNeXt Bottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(FasterNetBlock(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
