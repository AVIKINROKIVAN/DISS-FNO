import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from skimage.metrics import structural_similarity as ssim
from typing import Optional

class LpLoss:
    def __init__(self, d=2, p=2, reduce_dims=None):
        self.d = d  # Размерность пространства
        self.p = p  # Порядок нормы
        self.reduce_dims = reduce_dims  # По каким измерениям усреднять

    def __call__(self, pred, target, relative=False):
        # Вычисление ошибки в норме L^p
        err = torch.abs(pred - target)**self.p
        
        if self.reduce_dims is not None:
            err = torch.mean(err, dim=self.reduce_dims)
        
        loss = torch.mean(err)**(1/self.p)
        
        if relative:
            norm = torch.abs(target)**self.p
            if self.reduce_dims is not None:
                norm = torch.mean(norm, dim=self.reduce_dims)
            norm = torch.mean(norm)**(1/self.p)
            return loss / (norm + 1e-8)
        return loss

class MixedLoss:
    def __init__(self, alpha=0.7, l1_weight=1.0, l2_weight=1.0):
        self.alpha = alpha  # Вес для L1
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def __call__(self, pred, target):
        l1_loss = torch.abs(pred - target).mean() * self.l1_weight
        l2_loss = torch.square(pred - target).mean() * self.l2_weight
        return self.alpha * l1_loss + (1 - self.alpha) * l2_loss

def compute_ssim(pred, target, data_range=None):
    """Вычисление Structural Similarity Index (SSIM)"""
    if data_range is None:
        data_range = target.max() - target.min()
    return ssim(
        target.squeeze().numpy(),
        pred.squeeze().numpy(),
        data_range=data_range,
        win_size=11,
        channel_axis=None
    )

def plot_metrics(self):
    plt.figure(figsize=(12, 6))
    
    # График Loss
    plt.subplot(1, 2, 1)
    plt.plot(self.train_history['loss'], label='Train Loss')
    plt.plot(self.val_history['loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # График L2 Error
    plt.subplot(1, 2, 2)
    plt.plot(self.train_history['l2'], label='Train L2')
    plt.plot(self.val_history['l2'], label='Val L2')
    plt.title('L2 Error')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(self.save_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_solution_comparison(source, prediction, target, epoch=None, idx=0, save_path=None):
    """Сравнение решений с улучшенной визуализацией"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Вычисление общих границ цветовой шкалы
    vmin = min(np.min(source[idx]), np.min(prediction[idx]), np.min(target[idx]))
    vmax = max(np.max(source[idx]), np.max(prediction[idx]), np.max(target[idx]))
    
    # Разница между предсказанием и истинным решением
    diff = np.abs(prediction[idx] - target[idx])
    
    titles = [
        f'Source Term (min={source[idx].min():.2f}, max={source[idx].max():.2f})',
        f'Prediction (min={prediction[idx].min():.2f}, max={prediction[idx].max():.2f})',
        f'True Solution (min={target[idx].min():.2f}, max={target[idx].max():.2f})'
    ]
    
    data = [source[idx,...,0], prediction[idx,...,0], target[idx,...,0]]
    
    for ax, title, d in zip(axes, titles, data):
        im = ax.imshow(d, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    
    # Добавление информации об ошибке
    ssim_val = compute_ssim(prediction[idx], target[idx], data_range=vmax-vmin)
    l2_error = np.sqrt(np.mean(diff**2))
    
    suptitle = f'Max diff: {diff.max():.2e} | L2: {l2_error:.2e} | SSIM: {ssim_val:.3f}'
    if epoch is not None:
        suptitle = f'Epoch {epoch} - ' + suptitle
    
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Дополнительный график ошибки
    plt.figure(figsize=(6, 5))
    im = plt.imshow(diff, cmap='hot')
    plt.title(f'Absolute Error (Max: {diff.max():.2e})', fontsize=12)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_error.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(errors, save_path=None):
    """Визуализация распределения ошибок"""
    plt.figure(figsize=(10, 6))
    
    # Гистограмма
    plt.subplot(1, 2, 1)
    plt.hist(errors.flatten(), bins=50, density=True, alpha=0.7, color='b')
    plt.title('Error Distribution', fontsize=12)
    plt.xlabel('Absolute Error', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # QQ-plot для проверки нормальности
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(errors.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()