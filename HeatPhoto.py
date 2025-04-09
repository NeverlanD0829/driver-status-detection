import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layers, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()
        self.model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, self.model.state_dict(), exclude=['anchor'])
        self.model.load_state_dict(csd, strict=False)
        self.model.eval()
        print(f'Transferred {len(csd)}/{len(self.model.state_dict())} items')
        
        # Correct access to target layers
        self.target_layers = [getattr(self.model.model, f'{layer}') for layer in layers]
        self.method = eval(method)
        self.device = device
        self.backward_type = backward_type
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.model_names = model_names

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)
        result = grads(tensor)
        activations = [grads.activations[i].cpu().detach().numpy() for i in range(len(self.target_layers))]
        
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf_threshold:
                break
            
            self.model.zero_grad()
            score = post_result[i].max()
            score.backward(retain_graph=True)
            
            # 初始化 final_heatmap 与原始图像大小一致
            final_heatmap = np.zeros((img.shape[0], img.shape[1]))
            
            for j, act in enumerate(activations):
                gradients = grads.gradients[j]
                weights = self.method.get_cam_weights(self.method, None, None, None, act, gradients.detach().numpy())
                saliency_map = np.sum(weights.reshape(weights.shape[0], weights.shape[1], 1, 1) * act, axis=1)
                saliency_map = np.squeeze(np.maximum(saliency_map, 0))
                
                # 将 saliency_map 调整为与原始图像一致的大小 (640x640)
                saliency_map_resized = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))
                
                # 将调整后的 saliency_map 加到 final_heatmap 上
                final_heatmap += saliency_map_resized
            
            # 对 final_heatmap 进行归一化处理
            final_heatmap = (final_heatmap - final_heatmap.min()) / (final_heatmap.max() - final_heatmap.min())

            # 将 final_heatmap 应用到原始图像上
            cam_image = show_cam_on_image(img.copy(), final_heatmap, use_rgb=True)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')



def get_params():
    params = {
        'weight': '/home/chen/Desktop/yolo-V8/runs/detect/v8s-p2/weights/best.pt',  # 训练出来的权重文件
        'cfg': 'ultralytics/cfg/models/v8/yolov8s-p2.yaml',  # 训练权重对应的yaml配置文件


        # 'weight': '/home/chen/Desktop/yolo-V8/runs/detect/yolov8s-C2fiEMA-inMPDIOU-AFPN/weights/best.pt',  # 训练出来的权重文件
        # 'cfg': '/home/chen/Desktop/yolo-V8/ultralytics/cfg/models/v8/yolov8s-C2fiEMA-inMPDIOU-AFPN.yaml',  # 训练权重对应的yaml配置文件

        # 'weight': '/home/chen/Desktop/yolo-V8/runs/detect/v8s-coor/weights/best.pt',  # 训练出来的权重文件
        # 'cfg': '/home/chen/Desktop/yolo-V8/ultralytics/cfg/models/v8/yolov8s-attention.yaml',  # 训练权重对应的yaml配置文件

        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',  # GradCAMPlusPlus, GradCAM, XGradCAM , 使用的热力图库文件不同的效果不一样可以多尝试
        # 'layers': [9],  # 想要检测的对应层
        'layers': [2,4,6,8],  # 想要检测的对应层
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.6,  # 0.6  # 置信度阈值，有的时候你的进度条到一半就停止了就是因为没有高于此值的了
        'ratio': 0.02  # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    model(r'/home/chen/Desktop/大论文相关/大论文插图/origin.jpg', '/home/chen/Desktop/大论文相关/大论文插图/实验3：不同注意力机制/origin/Without_Att')  # 第一个是检测的文件, 第二个是保存的路径
