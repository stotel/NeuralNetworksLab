import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import RTDETR
import matplotlib.pyplot as plt
import math


MODEL_PATH = '../runs/detect/rtdetr_tuned/weights/best.pt'

TARGET_LAYER_IDX = 21

FILTER_INDICES = np.random.choice(range(256), size=50, replace=False).tolist()
STEPS = 150
LR = 0.05


class RTDETRActivationMaximization:
    def __init__(self, model_path, layer_idx):
        self.model = RTDETR(model_path)
        self.model_pt = self.model.model.float().eval()

        # Вимикаємо градієнти для ваг моделі
        for param in self.model_pt.parameters():
            param.requires_grad = False

        try:
            self.target_layer = self.model_pt.model[layer_idx]
        except:
            raise ValueError(f"Layer {layer_idx} not found. Check model architecture.")

        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output

    def random_jitter(self, img, max_jitter=16):
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        return torch.roll(torch.roll(img, ox, -1), oy, -2)

    def generate(self, filter_idx=0, steps=50, lr=0.05, combine_all=False, octaves=3, octave_scale=1.4):
        final_h, final_w = 640, 640

        def align_to_32(size):
            return int(round(size / 32) * 32)

        raw_start_h = final_h / (octave_scale ** (octaves - 1))
        raw_start_w = final_w / (octave_scale ** (octaves - 1))

        start_h = max(32, align_to_32(raw_start_h))
        start_w = max(32, align_to_32(raw_start_w))

        img_tensor = torch.from_numpy(np.random.uniform(100, 160, (1, 3, start_h, start_w)).astype(np.uint8))
        img_tensor = img_tensor.float().div(255.0).permute(0, 1, 2, 3)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_pt.to(device)
        img_tensor = img_tensor.to(device)

        mode_str = "ALL FILTERS" if combine_all else f"Filter {filter_idx}"

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
            current_jitter = 16 + octave * 8

            for i in range(steps):
                optimizer.zero_grad()
                jittered_img = self.random_jitter(img_tensor, max_jitter=current_jitter)

                # Forward RT-DETR
                _ = self.model_pt(jittered_img)
                layer_output = self.activations


                if isinstance(layer_output, list):
                    target_act = layer_output[-1]
                else:
                    target_act = layer_output

                if combine_all:
                    loss = -torch.mean(target_act)
                else:
                    # Перевірка на кількість каналів
                    if filter_idx >= target_act.shape[1]:
                        print(
                            f"Warning: Filter index {filter_idx} out of bounds for layer with {target_act.shape[1]} channels. Skipping.")
                        return np.zeros((640, 640, 3), dtype=np.uint8)
                    loss = -torch.mean(target_act[0, filter_idx])

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
    try:
        visualizer = RTDETRActivationMaximization(MODEL_PATH, TARGET_LAYER_IDX)

        images_to_show = []

        deep_dream_img = visualizer.generate(steps=STEPS, lr=LR, combine_all=True, octaves=6, octave_scale=1.2)
        images_to_show.append(("RT-DETR Deep Dream", deep_dream_img))

        total_images = len(images_to_show)
        cols = 3
        rows = math.ceil(total_images / cols)

        plt.figure(figsize=(5 * cols, 5 * rows))
        for i, (title, img) in enumerate(images_to_show):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            safe_title = title.replace(" ", "_")
            cv2.imwrite(f'rtdetr_actmax_{safe_title}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        plt.tight_layout()
        plt.show()
        print("\nRT-DETR Gallery generated.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()