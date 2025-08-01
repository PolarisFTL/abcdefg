import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from thop import clever_format, profile
from model import YoloBody


# ==============================
# THE RESULTS OF THE OUR SSAN
# Total GFLOPS: 21.307G
# Total params: 6.663M
# ==============================

if __name__ == "__main__":
    input_shape     = [640, 640]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 5
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(anchors_mask, num_classes, False).to(device)
    for i in m.children():
        print(i)
        print('==============================')
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)

    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
