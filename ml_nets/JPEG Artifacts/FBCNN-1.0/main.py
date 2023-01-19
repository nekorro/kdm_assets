import os.path
import sys
import torch
from utils import utils_image as util

n_channels = 3            # set 1 for grayscale image, set 3 for color image
model_name = 'fbcnn_color.pth'
nc = [64,128,256,512]
nb = 4   
    
L_path = sys.argv[1]
E_path = sys.argv[1]   # E_path, for Estimated images

model_pool = 'model_zoo'  # fixed
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_pool, model_name)
print(os.path.dirname(os.path.realpath(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------
# load model
# ----------------------------------------

from models.network_fbcnn import FBCNN as net
model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
print('Model path: {:s}'.format(model_path))

L_paths = util.get_image_paths(L_path)
for idx, img in enumerate(L_paths):

    # ------------------------------------
    # (1) img_L
    # ------------------------------------
    img_name, ext = os.path.splitext(os.path.basename(img))
    print('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
    img_L = util.imread_uint(img, n_channels=n_channels)
       
    img_L = util.uint2tensor4(img_L)
    img_L = img_L.to(device)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------
    img_E, QF = model(img_L)
    QF = 1 - QF
    img_E = util.tensor2single(img_E)
    img_E = util.single2uint(img_E)
    print('predicted quality factor: {:d}'.format(round(float(QF*100))))
    util.imsave(img_E, os.path.join(E_path, img_name+'.png'))