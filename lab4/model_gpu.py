import cupy as cp
import sys

class NeuralNetMLP_CP:
    def __init__(self, n_hidden:int = 30,
                 l2:float = 0.,
                 epochs:int = 100,
                 eta:float = 0.001,
                 shuffle:bool = True,
                 minibatch_size:int = 1,
                 seed:int|None = None
                 ):
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
            "train_acc": [],
            "test_acc": [],
        }
        
    def sigmoid(self, x):
        return cp.clip(1 / (1 + cp.exp(-x)), 1e-7, 1-1e-7)
    
    def _onehot(self, y, n_classes):
        onehot = cp.zeros((len(y), n_classes))
        onehot[cp.arange(len(y)), y] = 1
        return onehot
    
    def softmax(self, x):
        e_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
        return e_x / cp.sum(e_x, axis=1, keepdims=True)
    
    def _forward(self, X):
        z_h = cp.dot(X, self.w_h) + self.b_h
        a_h = self.sigmoid(z_h)
        z_out = cp.dot(a_h, self.w_out) + self.b_out
        a_out = self.softmax(z_out)
        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_tr, output):
        L2_term = (self.l2 * (cp.sum(self.w_h ** 2.) + cp.sum(self.w_out ** 2.)))
        term1 = y_tr * (cp.log(output+1e-7))
        term2 = (1. - y_tr) * (cp.log(1. - output + 1e-7))  # Added small constant for numerical stability
        cost = cp.sum(term1-term2) + L2_term
        return cost
    
    def predict(self, X):
        z_h, a_h, z_out, a_out = self._forward(X)
        return cp.argmax(a_out, axis=1)
    
    def fit(self, X_train, y_train, X_test, y_test):
        # Ensure inputs are CuPy arrays (no conversion back to NumPy)
        X_train = cp.asarray(X_train)
        y_train = cp.asarray(y_train)
        X_test = cp.asarray(X_test)
        y_test = cp.asarray(y_test)
        
        assert X_train.ndim == 2, f"X_train must be 2D, got {X_train.shape}"
        assert y_train.ndim == 1, f"y_train must be 1D, got {y_train.shape}"
        
        if self.minibatch_size <= 0:
            self.minibatch_size = X_train.shape[0]
        
        n_output = int(cp.max(y_train)) + 1  # Dynamically determine number of classes
        n_features = X_train.shape[1]
        
        # Initialize weights
        scale_h = cp.sqrt(2 / (n_features + self.n_hidden))
        scale_out = cp.sqrt(2 / (self.n_hidden + n_output))
        self.b_h = cp.zeros(self.n_hidden)
        self.b_out = cp.zeros(n_output)
        self.w_h = self.random.normal(loc=0.0, scale=scale_h, size=(n_features, self.n_hidden))
        self.w_out = self.random.normal(loc=0.0, scale=scale_out, size=(self.n_hidden, n_output))
        
        y_train_enc = self._onehot(y_train, n_output)
        
        for epoch in range(self.epochs):
            indices = cp.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
                
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                X_batch = X_train[batch_idx]
                y_batch_enc = y_train_enc[batch_idx]
                
                # Forward pass
                z_h, a_h, z_out, a_out = self._forward(X_batch)
                
                # Backpropagation
                delta_out = a_out - y_batch_enc
                sigmoid_der_h = a_h * (1. - a_h)
                delta_h = (cp.dot(delta_out, self.w_out.T) * sigmoid_der_h)
                
                # Weight updates
                self.w_out -= self.eta * (cp.dot(a_h.T, delta_out) + self.l2*self.w_out)
                self.b_out -= self.eta * cp.sum(delta_out, axis=0)
                self.w_h -= self.eta * (cp.dot(X_batch.T, delta_h) + self.l2*self.w_h)
                self.b_h -= self.eta * cp.sum(delta_h, axis=0)
            
            # Compute metrics (all in CuPy)
            _, _, _, a_out = self._forward(X_train)
            cost = self._compute_cost(y_train_enc, a_out)
            
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)
            
            train_acc = cp.mean(y_train == y_train_pred)
            test_acc = cp.mean(y_test == y_test_pred)
            
            # Store metrics (convert to float only at the end)
            self.cost.append(float(cost))
            self.eval_['train_acc'].append(float(train_acc))
            self.eval_['test_acc'].append(float(test_acc))
            
            sys.stderr.write(
                f"\rEpoch {epoch+1}/{self.epochs} | "
                f"Cost: {float(cost):.2f} | "
                f"Train/Test Acc.: {float(train_acc)*100:.2f}%/{float(test_acc)*100:.2f}%"
            )
            sys.stderr.flush()
            
        return self