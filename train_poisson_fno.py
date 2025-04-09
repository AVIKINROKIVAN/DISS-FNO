import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import scipy.io as sio
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from fno_model import FNO2d
from utilities import MixedLoss
from scipy.ndimage import zoom

class PoissonSolver:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Увеличиваем weight decay для более сильной регуляризации
        self.model = FNO2d(
            modes1=config.modes,
            modes2=config.modes,
            width=config.width
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=1e-3  # Увеличено с 1e-4
        )
        
        # Более агрессивное снижение LR
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5
        )
        
        self.loss_fn = nn.MSELoss()
        self.mixed_loss = MixedLoss(alpha=0.7)
        
        self.best_loss = float('inf')
        self.train_history = {'loss': [], 'l2': []}
        self.val_history = {'loss': [], 'l2': []}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(config.save_dir, f"run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

    def resize_to_power_of_two(self, data, original_size, target_size=64):
        """Изменяет размер данных до степени двойки"""
        return zoom(data.reshape(-1, original_size, original_size, 1),
                   (1, target_size/original_size, target_size/original_size, 1),
                   order=1)

    def load_and_preprocess_data(self):
        try:
            data = sio.loadmat(self.config.data_path)
            sources = data['source_data'].astype(np.float32)
            solutions = data['solution_data'].astype(np.float32)
            
            original_size = int(np.sqrt(sources.shape[1]))
            target_size = 64
            
            # Изменение размера до степени двойки
            sources = self.resize_to_power_of_two(sources, original_size, target_size)
            solutions = self.resize_to_power_of_two(solutions, original_size, target_size)
            
            # Нормализация
            self.sources_mean, self.sources_std = sources.mean(), sources.std()
            self.solutions_mean, self.solutions_std = solutions.mean(), solutions.std()
            
            sources = (sources - self.sources_mean) / (self.sources_std + 1e-8)
            solutions = (solutions - self.solutions_mean) / (self.solutions_std + 1e-8)
            
            # Преобразование в тензоры
            sources = torch.FloatTensor(sources)
            solutions = torch.FloatTensor(solutions)
            
            # Разделение данных
            dataset = TensorDataset(sources, solutions)
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            self.train_data, self.val_data, self.test_data = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            self.train_loader = DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=True
            )
            self.val_loader = DataLoader(
                self.val_data,
                batch_size=self.config.batch_size,
                pin_memory=True
            )
            self.test_loader = DataLoader(
                self.test_data,
                batch_size=self.config.batch_size,
                pin_memory=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_l2 = 0.0
        
        for x, y in tqdm(self.train_loader, desc="Training", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(x)
            loss = self.mixed_loss(outputs, y)
            l2_loss = torch.mean((outputs - y)**2)
            
            loss.backward()
            
            # Gradient Clipping с максимальной нормой 0.5
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_l2 += l2_loss.item()
        
        return total_loss / len(self.train_loader), total_l2 / len(self.train_loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_l2 = 0.0
        
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Validating", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.mixed_loss(outputs, y)
                l2_loss = torch.mean((outputs - y)**2)
                total_loss += loss.item()
                total_l2 += l2_loss.item()
        
        return total_loss / len(loader), total_l2 / len(loader)

    def visualize_results(self, epoch):
        self.model.eval()
        with torch.no_grad():
            x, y = next(iter(self.val_loader))
            x, y = x[:1].to(self.device), y[:1].to(self.device)
            
            pred = self.model(x).cpu()
            
            # Денормализация
            x = x.cpu() * self.sources_std + self.sources_mean
            y = y.cpu() * self.solutions_std + self.solutions_mean
            pred = pred * self.solutions_std + self.solutions_mean
            
            # Визуализация
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            titles = ['Input', 'Prediction', 'True Solution']
            data = [x.squeeze().numpy(), pred.squeeze().numpy(), y.squeeze().numpy()]
            
            vmin = min(d.min() for d in data)
            vmax = max(d.max() for d in data)
            
            for ax, title, d in zip(axes, titles, data):
                im = ax.imshow(d, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(title)
                fig.colorbar(im, ax=ax)
                ax.axis('off')
            
            error = np.abs(pred.squeeze().numpy() - y.squeeze().numpy())
            plt.suptitle(f'Epoch {epoch} - Max Error: {error.max():.2e}', fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(self.save_dir, f'epoch_{epoch:04d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # График ошибки
            plt.figure(figsize=(6, 5))
            plt.imshow(error, cmap='hot')
            plt.title(f'Absolute Error (Max: {error.max():.2e})')
            plt.colorbar()
            plt.axis('off')
            plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch:04d}_error.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_metrics(self):
        plt.figure(figsize=(15, 6))
        
        # График Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['loss'], 'b-', label='Train Loss')
        plt.plot(self.val_history['loss'], 'r-', label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # График L2 Error
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['l2'], 'b--', label='Train L2')
        plt.plot(self.val_history['l2'], 'r--', label='Val L2')
        plt.title('L2 Error')
        plt.xlabel('Epoch')
        plt.ylabel('L2 Error')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def train(self):
        try:
            self.load_and_preprocess_data()
            
            print("\nModel architecture:")
            print(self.model)
            print(f"\nTotal parameters: {sum(p.numel() for p in self.model.parameters())}")
            
            for epoch in range(1, self.config.epochs + 1):
                start_time = time.time()
                
                train_loss, train_l2 = self.train_epoch()
                val_loss, val_l2 = self.validate(self.val_loader)
                
                self.train_history['loss'].append(train_loss)
                self.train_history['l2'].append(train_l2)
                self.val_history['loss'].append(val_loss)
                self.val_history['l2'].append(val_l2)
                
                self.scheduler.step(val_loss)
                
                # Сохраняем модель только если val_loss улучшился
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                    print(f"New best model saved at epoch {epoch} with val_loss {val_loss:.4e}")
                
                if epoch % self.config.print_freq == 0 or epoch == 1:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"\nEpoch {epoch:04d}/{self.config.epochs} | "
                          f"LR: {current_lr:.1e} | "
                          f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | "
                          f"Train L2: {train_l2:.4e} | Val L2: {val_l2:.4e} | "
                          f"Time: {time.time()-start_time:.1f}s")
                
                if epoch % self.config.viz_freq == 0 or epoch == 1:
                    self.visualize_results(epoch)
                    self.plot_metrics()
            
            test_loss, test_l2 = self.validate(self.test_loader)
            print(f"\nTraining completed. Best val loss: {self.best_loss:.4e}")
            print(f"Test Loss: {test_loss:.4e} | Test L2: {test_l2:.4e}")
            
        except Exception as e:
            print(f"\nTraining failed: {str(e)}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='FNO for Poisson Equation')
    parser.add_argument('--data_path', required=True, help='Path to .mat dataset')
    parser.add_argument('--save_dir', default='results', help='Output directory')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes')  # Уменьшено с 16
    parser.add_argument('--width', type=int, default=48, help='Model width')  # Уменьшено с 64
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')  # Уменьшено с 500
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--print_freq', type=int, default=5, help='Log frequency')
    parser.add_argument('--viz_freq', type=int, default=10, help='Visualization frequency')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    solver = PoissonSolver(args)
    solver.train()