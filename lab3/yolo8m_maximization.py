import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math

MODEL_PATH = '../runs/detect/yolov8_tuned/weights/best.pt'
TARGET_LAYER_IDX = 18

FILTER_INDICES = np.random.choice(range(256), size=15, replace=False).tolist()
STEPS = 150
LR = 0.05


class ActivationMaximization:
    def __init__(self, model_path, layer_idx):
        self.model = YOLO(model_path)
        self.model_pt = self.model.model.float().eval()

        for param in self.model_pt.parameters():
            param.requires_grad = False

        try:
            self.target_layer = self.model_pt.model[layer_idx]
            print(f"Target Layer found: {self.target_layer}")
        except:
            raise ValueError(f"Layer {layer_idx} not found.")

        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output

    def random_jitter(self, img, max_jitter=16):
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        return torch.roll(torch.roll(img, ox, -1), oy, -2)

    def generate(self, filter_idx=0, steps=50, lr=0.05, combine_all=False, octaves=3, octave_scale=1.1):
        final_h, final_w = 640, 640

        def align_to_32(size):
            return int(round(size / 32) * 32)

        raw_start_h = final_h / (octave_scale ** (octaves - 1))
        raw_start_w = final_w / (octave_scale ** (octaves - 1))

        start_h = align_to_32(raw_start_h)
        start_w = align_to_32(raw_start_w)

        start_h = max(32, start_h)
        start_w = max(32, start_w)

        img_tensor = torch.from_numpy(np.random.uniform(100, 160, (1, 3, start_h, start_w)).astype(np.uint8))
        img_tensor = img_tensor.float().div(255.0).permute(0, 1, 2, 3)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_pt.to(device)
        img_tensor = img_tensor.to(device)


        for octave in range(octaves):

            raw_new_h = start_h * (octave_scale ** octave)
            raw_new_w = start_w * (octave_scale ** octave)


            new_h = align_to_32(raw_new_h)
            new_w = align_to_32(raw_new_w)


            if octave == octaves - 1:
                new_h, new_w = final_h, final_w

            if octave > 0:
                img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

            img_tensor = img_tensor.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([img_tensor], lr=lr)

            print(f"  Octave {octave + 1}/{octaves} (Size: {new_w}x{new_h})")

            for i in range(steps):
                optimizer.zero_grad()

                jittered_img = self.random_jitter(img_tensor, max_jitter=20 + octave * 4)

                _ = self.model_pt(jittered_img)
                layer_output = self.activations

                if combine_all:
                    loss = -torch.mean(layer_output)
                else:
                    loss = -torch.mean(layer_output[0, filter_idx])

                loss.backward()

                if img_tensor.grad is not None:
                    img_tensor.grad /= (torch.mean(torch.abs(img_tensor.grad)) + 1e-8)

                optimizer.step()

                with torch.no_grad():
                    img_tensor.data.clamp_(0, 1)

        result_img = img_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())
        result_img = (result_img * 255).astype(np.uint8)

        return result_img


if __name__ == '__main__':
    visualizer = ActivationMaximization(MODEL_PATH, TARGET_LAYER_IDX)

    images_to_show = []

    deep_dream_img = visualizer.generate(steps=STEPS, lr=LR, combine_all=True, octaves=10)
    images_to_show.append(("Deep Dream", deep_dream_img))


    total_images = len(images_to_show)
    cols = 4
    rows = math.ceil(total_images / cols)

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, (title, img) in enumerate(images_to_show):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

        # Зберігаємо файл
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
        cv2.imwrite(f'activation_max_{safe_title}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    plt.show()
    print("\nGallery generated.")