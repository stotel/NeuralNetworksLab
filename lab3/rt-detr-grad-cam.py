import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import RTDETR
import traceback

if __name__ == '__main__':

    class RTDETRGradCAM:
        def __init__(self, model_path, target_layer_name=None):

            self.model = RTDETR(model_path)
            self.model_pt = self.model.model.float().eval()

            for param in self.model_pt.parameters():
                param.requires_grad = True

            self.target_layer = None

            if target_layer_name is not None:
                self.target_layer = self.model_pt.model[target_layer_name]
            else:
                try:
                    self.target_layer = self.model_pt.model[4]
                    print("Auto-selected target layer: model.model[4] (Backbone output)")
                except:
                    print("Could not auto-select layer 4, falling back to layer 0")
                    self.target_layer = self.model_pt.model[0]

            self.activations = None
            self.gradients = None
            self._register_hooks()
            self.names = self.model.names

        def _register_hooks(self):
            self.target_layer.register_forward_hook(self._save_activation)
            self.target_layer.register_full_backward_hook(self._save_gradient)

        def _save_activation(self, module, input, output):
            self.activations = output

        def _save_gradient(self, module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def preprocess_image(self, img_path, img_size=640):
            original_img = cv2.imread(img_path)
            if original_img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            resized_img = cv2.resize(original_img, (img_size, img_size))
            img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            return tensor_img, original_img, resized_img

        def get_grad_cam(self, input_tensor, target_class_idx=0):
            input_tensor.requires_grad = True

            pred = self.model_pt(input_tensor)

            if isinstance(pred, tuple):
                pred_output = pred[0]
            else:
                pred_output = pred

            self.model_pt.zero_grad()
            print(f"Prediction Output Shape: {pred_output.shape}")
            # Очікуємо [1, 300, 4 + NC] (або схоже)



            if len(pred_output.shape) == 3:
                # [Batch, Queries, Data]
                # RT-DETR output format: [cx, cy, w, h, class0, class1...]
                target_output = pred_output[0, :, 4 + target_class_idx]
            else:
                raise RuntimeError(f"Unexpected output shape: {pred_output.shape}")

            max_score = torch.max(target_output)
            print(f"Max score for class '{self.names[target_class_idx]}': {max_score.item()}")

            if not max_score.requires_grad:
                raise RuntimeError("Output tensor does not require grad.")


            max_score.backward(retain_graph=True)


            if self.gradients is None or self.activations is None:
                raise RuntimeError("No gradients/activations. Layer might not be in the forward pass.")

            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze()
            cam = torch.relu(cam)
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-8)

            return cam.detach().cpu().numpy()


    def visualize_cam(heatmap, original_img):
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)



    MODEL_PATH = '../runs/detect/rtdetr_tuned/weights/best.pt'
    IMAGE_PATH = '../dataset/images/1200px-KA_Zoo_Huftieranlage_jpg.rf.02958f89ec8def97694f66350d7fe199.jpg'
    OUTPUT_PATH = 'rtdetr_grad_cam.jpg'


    TARGET_CLASS_IDX = 10

    try:
        gradcam = RTDETRGradCAM(MODEL_PATH, target_layer_name=21)

        input_tensor, original_img, _ = gradcam.preprocess_image(IMAGE_PATH)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gradcam.model_pt.to(device)
        input_tensor = input_tensor.to(device)

        print(f"Calculating Grad-CAM for {gradcam.names[TARGET_CLASS_IDX]}...")
        heatmap = gradcam.get_grad_cam(input_tensor, target_class_idx=TARGET_CLASS_IDX)

        result = visualize_cam(heatmap, original_img)
        cv2.imwrite(OUTPUT_PATH, result)
        print(f"Saved to {OUTPUT_PATH}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()