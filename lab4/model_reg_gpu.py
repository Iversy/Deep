import cupy as cp
import sys
import time

class NeuralNetMLP_Regressor:
    def __init__(self, n_hidden=30,
                 l2=0.,
                 epochs=100,
                 eta=0.001,
                 shuffle=True,
                 minibatch_size=1,
                 seed=None):
        self.random = cp.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.w_h = None
        self.w_out = None
        self.b_h = None
        self.b_out = None
        self.cost = []
        self.eval_ = {
            "train_mse": [],
            "val_mse": [],
            "train_mae": [],
            "val_mae": []
        }
        
    def sigmoid(self, x):
        return cp.clip(1 / (1 + cp.exp(-x)), 1e-7, 1-1e-7)
    
    def _forward(self, X):
        z_h = cp.dot(X, self.w_h) + self.b_h
        a_h = self.sigmoid(z_h)
        z_out = cp.dot(a_h, self.w_out) + self.b_out
        return z_h, a_h, z_out
    
    def _compute_cost(self, y_true, y_pred):
        mse = cp.mean((y_true - y_pred)**2)
        L2_term = (self.l2 * (cp.sum(self.w_h**2) + cp.sum(self.w_out**2)))
        return mse + L2_term
    
    def predict(self, X):
        _, _, z_out = self._forward(X)
        return z_out
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, quiet=False):
        X_train = cp.asarray(X_train)
        y_train = cp.asarray(y_train).reshape(-1, 1).ravel()  # Регрессия - выход 1D
        
        if X_val is not None:
            X_val = cp.asarray(X_val)
            y_val = cp.asarray(y_val).reshape(-1, 1)
        
        start_time = time.time()
        
        if self.minibatch_size <= 0:
            self.minibatch_size = X_train.shape[0]
        
        n_features = X_train.shape[1]
        n_output = 1  # Для регрессии один выход
        
        # Инициализация весов
        scale_h = cp.sqrt(2 / (n_features + self.n_hidden))
        scale_out = cp.sqrt(2 / (self.n_hidden + n_output))
        
        self.b_h = cp.zeros(self.n_hidden)
        self.b_out = cp.zeros(n_output)
        self.w_h = self.random.normal(loc=0.0, scale=scale_h, 
                                    size=(n_features, self.n_hidden))
        self.w_out = self.random.normal(loc=0.0, scale=scale_out, 
                                      size=(self.n_hidden, n_output))
        
        for epoch in range(self.epochs):
            indices = cp.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
                
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, 
                                 self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Forward pass
                z_h, a_h, z_out = self._forward(X_batch)
                
                # Backpropagation
                delta_out = z_out - y_batch  # Производная MSE
                sigmoid_der_h = a_h * (1. - a_h)
                delta_h = (cp.dot(delta_out, self.w_out.T) * sigmoid_der_h)
                
                # Weight updates
                self.w_out -= self.eta * (cp.dot(a_h.T, delta_out) + self.l2*self.w_out)
                self.b_out -= self.eta * cp.sum(delta_out, axis=0)
                self.w_h -= self.eta * (cp.dot(X_batch.T, delta_h) + self.l2*self.w_h)
                self.b_h -= self.eta * cp.sum(delta_h, axis=0)
            
            # Вычисление метрик
            _, _, z_out_train = self._forward(X_train)
            train_mse = float(cp.mean((y_train - z_out_train)**2))
            train_mae = float(cp.mean(cp.abs(y_train - z_out_train)))
            
            self.cost.append(train_mse)
            self.eval_['train_mse'].append(train_mse)
            self.eval_['train_mae'].append(train_mae)
            
            if X_val is not None:
                _, _, z_out_val = self._forward(X_val)
                val_mse = float(cp.mean((y_val - z_out_val)**2))
                val_mae = float(cp.mean(cp.abs(y_val - z_out_val)))
                self.eval_['val_mse'].append(val_mse)
                self.eval_['val_mae'].append(val_mae)
            
            if not quiet:
                sys.stderr.write(
                    f"\rEpoch {epoch+1}/{self.epochs} | "
                    f"MSE: {train_mse:.4f} | "
                    f"MAE: {train_mae:.4f}"
                )
                if X_val is not None:
                    sys.stderr.write(
                        f" | Val MSE: {val_mse:.4f} | "
                        f"Val MAE: {val_mae:.4f}"
                    )
                sys.stderr.flush()
        
        self.time = time.time() - start_time
        print(f"\nTotal training time: {self.time:.2f} seconds")
        return self