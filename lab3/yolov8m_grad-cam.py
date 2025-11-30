import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
import traceback

if __name__ == '__main__':

    class YOLOGradCAM:

        def __init__(self, model_path, target_layer_name=18):
            self.model = YOLO(model_path)
            self.model_pt = self.model.model.float().eval()


            for param in self.model_pt.parameters():
                param.requires_grad = True

            try:
                self.target_layer = self.model_pt.model[target_layer_name]
                print(f"Target layer set to: {self.target_layer._get_name()} (Index {target_layer_name})")
            except KeyError:
                print(f"Warning: Layer {target_layer_name} not found, defaulting to last layer")
                self.target_layer = list(self.model_pt.modules())[-2]

            self.activations = None
            self.gradients = None

            self._register_hooks()

            # Назви класів (для відображення)
            self.names = self.model.names

        def _register_hooks(self):
            """Реєстрація forward та backward хуків на цільовому шарі."""
            self.target_layer.register_forward_hook(self._save_activation)
            self.target_layer.register_full_backward_hook(self._save_gradient)

        def _save_activation(self, module, input, output):
            """Зберігає вихідні ознаки (активації)."""
            self.activations = output

        def _save_gradient(self, module, grad_in, grad_out):
            """Зберігає градієнти, що проходять назад."""
            self.gradients = grad_out[0]

        def preprocess_image(self, img_path, img_size=640):
            """Завантаження та підготовка зображення для моделі."""
            original_img = cv2.imread(img_path)
            if original_img is None:
                raise FileNotFoundError(f"Зображення не знайдено за шляхом: {img_path}")

            resized_img = cv2.resize(original_img, (img_size, img_size))
            img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

            return tensor_img, original_img, resized_img

        def get_grad_cam(self, input_tensor, target_class_idx=0):
            """Обчислення Grad-CAM для конкретного класу."""

            input_tensor.requires_grad = True

            pred = self.model_pt(input_tensor)


            if isinstance(pred, tuple):
                pred_output = pred[0]
            elif isinstance(pred, list):
                pred_output = pred[-1]
            else:
                pred_output = pred

            # 3. Обнулення градієнтів
            self.model_pt.zero_grad()

            print(f"Prediction Output Shape: {pred_output.shape}")
            print("for class:", self.model_pt.names[target_class_idx])

            target_output = pred_output[0, 4 + target_class_idx, :]

            max_score = torch.max(target_output)
            print(f"Max score for class {self.names[target_class_idx]}: {max_score.item()}")

            if not max_score.requires_grad:
                raise RuntimeError(
                    "Output tensor does not require grad. Model structure might have detached gradients.")

            max_score.backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("No gradients or activations captured. Check layer registration.")


            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)


            cam = torch.sum(weights * self.activations, dim=1).squeeze()


            cam = torch.relu(cam)


            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-8)

            return cam.detach().cpu().numpy()


    def visualize_cam(heatmap, original_img, resized_img):
        """Візуалізація Heatmap та накладання на зображення."""


        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

        return superimposed_img

    MODEL_PATH = '../runs/detect/yolov8_tuned/weights/best.pt'  # Виправлено шлях згідно вашого контексту


    IMAGE_PATH = '../dataset/images/1200px-KA_Zoo_Huftieranlage_jpg.rf.02958f89ec8def97694f66350d7fe199.jpg'
    # IMAGE_PATH = '../dataset/images/Ayrshirecattle2_c_jpg.rf.e6de6435f5f8936b39eb218e00669f41.jpg'
    # IMAGE_PATH = '../dataset/images/0174798_red-cow-for-qurbani-sraf002-removebg-preview_png_jpg.rf.c6b73246f48c43ecb99912408a8b7445.jpg'
    IMAGE_PATH = '../dataset/images/hf03-16_jpeg_jpg.rf.61f5a0115f6ef481b016bad1e7491b95.jpg'

    OUTPUT_PATH = 'yolov8_grad_cam_output.jpg'

    try:
        yolo_cam = YOLOGradCAM(MODEL_PATH)

        input_tensor, original_img, resized_img = yolo_cam.preprocess_image(IMAGE_PATH)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        yolo_cam.model_pt.to(device)

        print(f"Обчислення Grad-CAM для класу: {yolo_cam.names[0]}...")

        heatmap = yolo_cam.get_grad_cam(input_tensor, target_class_idx=0)

        cam_image = visualize_cam(heatmap, original_img, resized_img)


        cv2.imwrite(OUTPUT_PATH, cam_image)

        print(f"\nGrad-CAM візуалізація збережена як {OUTPUT_PATH}")

    except Exception as e:
        print(f"\nerror occurred: {e}")
        traceback.print_exc()