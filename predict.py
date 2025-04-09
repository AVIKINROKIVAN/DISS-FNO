import torch
import numpy as np
import matplotlib.pyplot as plt
from fno_model import FNO2d
from utilities import LpLoss
import scipy.io as sio
import argparse
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class PoissonPredictor:
    def __init__(self, checkpoint_path, device=None):
        """Инициализация с загрузкой чекпоинта"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.normalization = self.checkpoint['normalization']
        
        # Инициализация модели
        self.model = FNO2d(
            modes1=self.checkpoint.get('modes', 12),
            modes2=self.checkpoint.get('modes', 12),
            width=self.checkpoint.get('width', 32)
        ).to(self.device)
        
        # Загрузка весов
        self.model.load_state_dict(self.checkpoint['model_state'])
        self.model.eval()
        
        # Метрики
        self.loss_fn = LpLoss()
        print(f"Loaded model from {checkpoint_path} (val loss: {self.checkpoint['best_loss']:.4e})")

    def preprocess(self, data):
        """Нормализация входных данных"""
        return (data - self.normalization['sources_mean']) / (self.normalization['sources_std'] + 1e-8)
    
    def postprocess(self, data):
        """Денормализация выходных данных"""
        return data * self.normalization['solutions_std'] + self.normalization['solutions_mean']

    def load_data(self, data_path):
        """Загрузка и подготовка данных"""
        data = sio.loadmat(data_path)
        sources = data['source_data'].astype(np.float32)
        solutions = data.get('solution_data', np.zeros_like(sources)).astype(np.float32)
        
        # Нормализация
        sources = self.preprocess(sources)
        
        # Преобразование в тензоры
        grid_size = int(np.sqrt(sources.shape[1]))
        sources = sources.reshape(-1, grid_size, grid_size, 1)
        solutions = solutions.reshape(-1, grid_size, grid_size, 1)
        
        return (
            torch.FloatTensor(sources),
            torch.FloatTensor(solutions) if 'solution_data' in data else None,
            data.get('mesh_coords', None)
        )

    def predict(self, input_data):
        """Генерация предсказаний"""
        with torch.no_grad():
            input_data = input_data.to(self.device)
            predictions = self.model(input_data)
            return self.postprocess(predictions.cpu())

    def evaluate(self, predictions, true_solutions):
        """Оценка качества"""
        if true_solutions is None:
            return None, None
            
        true_solutions = self.postprocess(true_solutions)
        abs_error = self.loss_fn(predictions, true_solutions)
        rel_error = self.loss_fn(predictions, true_solutions, relative=True)
        
        print(f"\nEvaluation Metrics:")
        print(f"Absolute L2 Error: {abs_error.item():.4e}")
        print(f"Relative L2 Error: {rel_error.item():.4e}")
        
        return abs_error.item(), rel_error.item()

    def visualize_3d(self, sources, predictions, solutions, mesh_coords, save_path=None):
        """3D визуализация результатов"""
        if mesh_coords is None:
            print("Cannot create 3D plot: mesh coordinates not provided")
            return
            
        idx = np.random.randint(0, len(sources))
        grid_size = int(np.sqrt(mesh_coords.shape[0]))
        
        fig = plt.figure(figsize=(18, 6))
        
        # Подготовка данных
        x = mesh_coords[:, 0].reshape(grid_size, grid_size)
        y = mesh_coords[:, 1].reshape(grid_size, grid_size)
        src = sources[idx].reshape(grid_size, grid_size)
        pred = predictions[idx].reshape(grid_size, grid_size)
        sol = solutions[idx].reshape(grid_size, grid_size) if solutions is not None else None
        
        # Source term
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(x, y, src, cmap='viridis', edgecolor='none')
        ax1.set_title('Source Term')
        
        # Prediction
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(x, y, pred, cmap='viridis', edgecolor='none')
        ax2.set_title('Model Prediction')
        
        # True Solution (если есть)
        if sol is not None:
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.plot_surface(x, y, sol, cmap='viridis', edgecolor='none')
            ax3.set_title('True Solution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_2d(self, sources, predictions, solutions, save_path=None):
        """2D сравнение результатов"""
        idx = np.random.randint(0, len(sources))
        
        fig, axes = plt.subplots(1, 3 if solutions is not None else 2, figsize=(15, 5))
        
        # Подготовка данных
        src = sources[idx].squeeze()
        pred = predictions[idx].squeeze()
        sol = solutions[idx].squeeze() if solutions is not None else None
        
        # Определение общего масштаба цветов
        vmin = min(src.min(), pred.min(), sol.min() if sol is not None else np.inf)
        vmax = max(src.max(), pred.max(), sol.max() if sol is not None else -np.inf)
        
        # Source term
        im0 = axes[0].imshow(src, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Source Term')
        fig.colorbar(im0, ax=axes[0])
        
        # Prediction
        im1 = axes[1].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Model Prediction')
        fig.colorbar(im1, ax=axes[1])
        
        # True Solution (если есть)
        if sol is not None:
            im2 = axes[2].imshow(sol, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[2].set_title('True Solution')
            fig.colorbar(im2, ax=axes[2])
            
            # Добавляем карту ошибок
            fig2, ax = plt.subplots(figsize=(6, 5))
            error = np.abs(pred - sol)
            im = ax.imshow(error, cmap='hot')
            ax.set_title('Absolute Error')
            fig2.colorbar(im, ax=ax)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path.replace('.png', '_error.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_predictions(self, predictions, save_path):
        """Сохранение предсказаний"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'predictions': predictions,
            'normalization': self.normalization
        }, save_path)
        print(f"Predictions saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='FNO Poisson Equation Predictor')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to input data (.mat)')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--plot_3d', action='store_true', help='Create 3D plots')
    args = parser.parse_args()
    
    # Инициализация предсказателя
    predictor = PoissonPredictor(args.checkpoint)
    
    # Загрузка данных
    sources, solutions, mesh_coords = predictor.load_data(args.data)
    
    # Предсказание
    predictions = predictor.predict(sources)
    
    # Оценка (если есть истинные решения)
    abs_error, rel_error = predictor.evaluate(predictions, solutions)
    
    # Визуализация
    if args.visualize:
        if args.plot_3d and mesh_coords is not None:
            predictor.visualize_3d(
                sources.numpy(),
                predictions.numpy(),
                solutions.numpy() if solutions is not None else None,
                mesh_coords
            )
        else:
            predictor.visualize_2d(
                sources.numpy(),
                predictions.numpy(),
                solutions.numpy() if solutions is not None else None,
                os.path.join(os.path.dirname(args.output), 'prediction.png') if args.output else None
            )
    
    # Сохранение результатов
    if args.output:
        predictor.save_predictions(predictions, args.output)

if __name__ == "__main__":
    main()