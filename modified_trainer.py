# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image

# from networks.depth_decoder_HW4 import DepthDecoder2 
# from networks.resnet_encoder_HW4 import ResnetEncoder2
# from layers import disp_to_depth

from networks.resnet_encoder import ResnetEncoder  # Importing from resnet_encoder.py
from networks import DepthDecoder 
from layers import disp_to_depth

# Dataset class
class SynDroneDataset(Dataset):
    def __init__(self, root_dir, root_dir_depth, height, width, is_train=True, transform=None):
        self.root_dir = root_dir
        self.root_dir_depth = root_dir_depth
        self.height = height
        self.width = width
        self.is_train = is_train
        self.transform = transform or self.default_transforms()
        self.image_pairs = self._load_image_pairs()

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])

    def _load_image_pairs(self):
        image_pairs = []
        for height_folder in ["height20m", "height80m"]:
            rgb_folder = os.path.join(self.root_dir, height_folder, "rgb")
            depth_folder = os.path.join(self.root_dir_depth, height_folder, "depth")
            
            for rgb_image_name in os.listdir(rgb_folder):
                rgb_image_path = os.path.join(rgb_folder, rgb_image_name)
                depth_image_path = os.path.join(depth_folder, rgb_image_name.replace(".jpg", ".png"))
                
                if os.path.exists(depth_image_path):
                    image_pairs.append((rgb_image_path, depth_image_path))
                else:
                    print(f"Warning: Missing depth image for {rgb_image_path} at {depth_image_path}")
        
        print(f"Total valid image pairs found: {len(image_pairs)}")
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.image_pairs[idx]
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("L")

        if self.transform:
            rgb_image = self.transform(rgb_image)
            rgb_image = transforms.functional.normalize(rgb_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        depth_image = transforms.ToTensor()(depth_image)  # Convert depth map to tensor

        return {"color": rgb_image, "depth": depth_image}

# Trainer class
class Trainer:
    def __init__(self, options):
        self.opt = options
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.models = self.initialize_models()
        self.writer = SummaryWriter('runs/rgb_depth_data')
        self.train_loader = self.get_dataloader()
        self.model_optimizer = self.configure_optimizer()
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        print(f"Using device: {self.device}")

    def initialize_models(self):
        models = {}
        # dropout_rate = 0.5  # Set dropout rate
        models["encoder"] = ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained").to(self.device)

        # models["encoder"] = ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained", dropout_rate=dropout_rate).to(self.device)
        models["depth"] = DepthDecoder(models["encoder"].num_ch_enc, self.opt.scales).to(self.device)
        # dropout_rate = self.opt.dropout_rate
        # models["encoder"] = ResnetEncoder2(num_layers=self.opt.num_layers, pretrained=self.opt.weights_init == "pretrained", dropout_rate=dropout_rate).to(self.device)
        # models["depth"] = DepthDecoder2(models["encoder"].num_ch_enc, self.opt.scales, dropout_rate=dropout_rate).to(self.device)
        return models

    def get_dataloader(self):
        dataset = SynDroneDataset(self.opt.data_path, self.opt.data_path_depth, self.opt.height, self.opt.width, is_train=True)
        return DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers)

    def configure_optimizer(self):
        parameters = list(self.models["encoder"].parameters()) + list(self.models["depth"].parameters())
        return optim.Adam(parameters, lr=self.opt.learning_rate)

    def train(self):
        self.step = 0
        for epoch in range(self.opt.num_epochs):
            epoch_loss = 0
            epoch_steps = 0
            for batch_idx, inputs in enumerate(self.train_loader):
                outputs, losses = self.process_batch(inputs)
                loss = losses["loss"]

                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

                if batch_idx % self.opt.log_frequency == 0:
                    self.writer.add_scalar("Loss/train_batch", loss.item(), self.step)
                    self.step += 1

                if batch_idx % 100 == 0:
                    self.log_depth_predictions(inputs, outputs, batch_idx)

            avg_epoch_loss = epoch_loss / epoch_steps
            self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
            print(f"Epoch [{epoch+1}/{self.opt.num_epochs}], Avg Loss: {avg_epoch_loss:.4f}")
            self.writer.flush()

            if (epoch + 1) % self.opt.save_frequency == 0:
                self.save_model(epoch)

            self.model_lr_scheduler.step()

        self.writer.close()

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        features = self.models["encoder"](inputs["color"])
        outputs = self.models["depth"](features)

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def log_depth_predictions(self, inputs, outputs, batch_idx):
        pred_disp = outputs[("disp", 0)].detach().cpu().numpy()
        pred_depth = 1 / pred_disp

        gt_depth = inputs["depth"].clone().to(self.device)
        gt_depth = gt_depth / torch.max(gt_depth)
        gt_depth = gt_depth.cpu().numpy()

        for i in range(min(10, pred_depth.shape[0])):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(pred_depth[i, 0], cmap="magma")
            ax[0].set_title("Predicted Depth Map")
            ax[1].imshow(gt_depth[i, 0], cmap="magma")
            ax[1].set_title("Ground Truth Depth Map")
            plt.tight_layout()
            
            self.writer.add_figure(f"Depth Comparisons/Example {i + batch_idx * 10}", fig, self.step)
            plt.close(fig)

    def compute_losses(self, inputs, outputs):
        pred_disp = outputs[("disp", 0)]
        gt_depth = inputs["depth"].to(pred_disp.device)
        
        gt_depth_resized = F.interpolate(gt_depth, size=pred_disp.shape[2:], mode="bilinear", align_corners=False)
        
        loss = F.mse_loss(pred_disp, gt_depth_resized)
        return {"loss": loss}

    def save_model(self, epoch):
        save_folder = os.path.join(self.opt.log_dir, "models", f"weights_{epoch}")
        os.makedirs(save_folder, exist_ok=True)
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(save_folder, f"{model_name}.pth"))

# import os
# import time
# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# from PIL import Image

# from networks import ResnetEncoder, DepthDecoder  # Ensure these are defined in your project
# from layers import disp_to_depth  # Convert disparity to depth
# import torch.nn.functional as F


# class SynDroneDataset(Dataset):
#     def __init__(self, root_dir, root_dir_depth, height, width, is_train=True, transform=None):
#         self.root_dir = root_dir
#         self.root_dir_depth = root_dir_depth
#         self.height = height
#         self.width = width
#         self.is_train = is_train
#         self.transform = transform or self.default_transforms()
#         self.image_pairs = self._load_image_pairs()

#     def default_transforms(self):
#         return transforms.Compose([
#             transforms.Resize((self.height, self.width)),
#             transforms.ToTensor()
#             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def _load_image_pairs(self):
#         image_pairs = []
#         for height_folder in ["height20m", "height80m"]:
#             rgb_folder = os.path.join(self.root_dir, height_folder, "rgb")
#             depth_folder = os.path.join(self.root_dir_depth, height_folder, "depth")
            
#             for rgb_image_name in os.listdir(rgb_folder):
#                 rgb_image_path = os.path.join(rgb_folder, rgb_image_name)
#                 depth_image_path = os.path.join(depth_folder, rgb_image_name.replace(".jpg", ".png"))
                
#                 if os.path.exists(depth_image_path):
#                     image_pairs.append((rgb_image_path, depth_image_path))
#                 else:
#                     print(f"Warning: Missing depth image for {rgb_image_path} at {depth_image_path}")
        
#         print(f"Total valid image pairs found: {len(image_pairs)}")
#         return image_pairs

#     def __len__(self):
#         return len(self.image_pairs)

#     def __getitem__(self, idx):
#         rgb_path, depth_path = self.image_pairs[idx]
#         rgb_image = Image.open(rgb_path).convert("RGB")
#         depth_image = Image.open(depth_path).convert("L")

#         if self.transform:
#             rgb_image = self.transform(rgb_image)
#             rgb_image = transforms.functional.normalize(rgb_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
#         # depth_image = self.transform(depth_image)
#         depth_image = transforms.ToTensor()(depth_image)  # Convert to tensor


#         return {"color": rgb_image, "depth": depth_image}


# class Trainer:
#     def __init__(self, options):
#         self.opt = options
#         self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#         self.models = self.initialize_models()
#         self.writer = SummaryWriter('runs/rgb_depth_data')
#         self.train_loader = self.get_dataloader()
#         self.model_optimizer = self.configure_optimizer()
#         self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
#         print(f"Using device: {self.device}")

#     def initialize_models(self):
#         models = {}
#         models["encoder"] = ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained").to(self.device)
#         models["depth"] = DepthDecoder(models["encoder"].num_ch_enc, self.opt.scales).to(self.device)
#         return models

#     def get_dataloader(self):
#         dataset = SynDroneDataset(self.opt.data_path, self.opt.data_path_depth, self.opt.height, self.opt.width, is_train=True)
#         return DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers)

#     def configure_optimizer(self):
#         parameters = list(self.models["encoder"].parameters()) + list(self.models["depth"].parameters())
#         return optim.Adam(parameters, lr=self.opt.learning_rate)

#     def train(self):
#         self.step = 0
#         for epoch in range(self.opt.num_epochs):
#             epoch_loss = 0
#             epoch_steps = 0
#             for batch_idx, inputs in enumerate(self.train_loader):
#                 outputs, losses = self.process_batch(inputs)
#                 loss = losses["loss"]

#                 self.model_optimizer.zero_grad()
#                 loss.backward()
#                 self.model_optimizer.step()

#                 epoch_loss += loss.item()
#                 epoch_steps += 1

#                 if batch_idx % self.opt.log_frequency == 0:
#                     self.writer.add_scalar("Loss/train_batch", loss.item(), self.step)
#                     self.step += 1

#                 if batch_idx % 100 == 0:
#                     self.log_depth_predictions(inputs, outputs, batch_idx)

#             avg_epoch_loss = epoch_loss / epoch_steps
#             self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
#             print(f"Epoch [{epoch+1}/{self.opt.num_epochs}], Avg Loss: {avg_epoch_loss:.4f}")
#             self.writer.flush()

#             if (epoch + 1) % self.opt.save_frequency == 0:
#                 self.save_model(epoch)

#             self.model_lr_scheduler.step()

#         self.writer.close()

#     def process_batch(self, inputs):
#         for key, ipt in inputs.items():
#             inputs[key] = ipt.to(self.device)
        
#         features = self.models["encoder"](inputs["color"])
#         outputs = self.models["depth"](features)

#         losses = self.compute_losses(inputs, outputs)
#         return outputs, losses

#     def log_depth_predictions(self, inputs, outputs, batch_idx):
#         pred_disp = outputs[("disp", 0)].detach().cpu().numpy()
#         pred_depth = 1 / pred_disp

#         gt_depth = inputs["depth"].clone().to(self.device)
#         gt_depth = gt_depth / torch.max(gt_depth)
#         gt_depth = gt_depth.cpu().numpy()

#         for i in range(min(10, pred_depth.shape[0])):
#             fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#             ax[0].imshow(pred_depth[i, 0], cmap="magma")
#             ax[0].set_title("Predicted Depth Map")
#             ax[1].imshow(gt_depth[i, 0], cmap="magma")
#             ax[1].set_title("Ground Truth Depth Map")
#             plt.tight_layout()
            
#             self.writer.add_figure(f"Depth Comparisons/Example {i + batch_idx * 10}", fig, self.step)
#             plt.close(fig)

#     def compute_losses(self, inputs, outputs):
#         pred_disp = outputs[("disp", 0)]
#         gt_depth = inputs["depth"].to(pred_disp.device)
        
#         gt_depth_resized = F.interpolate(gt_depth, size=pred_disp.shape[2:], mode="bilinear", align_corners=False)
        
#         # loss = torch.nn.functional.mse_loss(pred_disp, gt_depth)
#         loss = F.mse_loss(pred_disp, gt_depth_resized)
#         return {"loss": loss}

#     def save_model(self, epoch):
#         save_folder = os.path.join(self.opt.log_dir, "models", f"weights_{epoch}")
#         os.makedirs(save_folder, exist_ok=True)
#         for model_name, model in self.models.items():
#             torch.save(model.state_dict(), os.path.join(save_folder, f"{model_name}.pth"))

