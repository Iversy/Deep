import cupy as cp
import numpy as np
import sys
class NeuralNetMLP:
    def __init__(self, n_hidden:int = 30,
                 l2:float = 0.,
                 epochs:int = 100,
                 eta:float = 0.001,
                 shuffle:bool = True,
                 minibatch_size:int = 1,
                 seed:int|None = None
                 ):
        self.random = np.random.RandomState(seed)
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
        return np.clip(1 / (1 + np.exp(-x)), 1e-7, 1-1e-7)
    
        
    def _onehot(self, y, n_classes):
        onehot = np.zeros((len(y), n_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def _forward(self, X):
        z_h = np.dot(X, self.w_h) + self.b_h
        a_h = self.sigmoid(z_h)
        z_out = np.dot(a_h, self.w_out) + self.b_out
        a_out = self.softmax(z_out)  # Change to softmax
        return z_h, a_h, z_out, a_out

    # def _compute_cost(self, y_tr, output):
    #     L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
    #     # Categorical cross-entropy
    #     loss = -np.sum(y_tr * np.log(output + 1e-7)) + L2_term
    #     return loss
        
    def _compute_cost(self, y_tr, output):
        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = y_tr * (np.log(output+1e-7))
        term2 = (1. - y_tr) * (np.log(1. - output))
        cost = np.sum(term1-term2) + L2_term
        return cost
    
    # def _forward(self, X):
    #     z_h = np.dot(X, self.w_h) + self.b_h
    #     a_h = self.sigmoid(z_h)
    #     z_out = np.dot(a_h , self.w_out) + self.b_out
    #     a_out = self.sigmoid(z_out)
    #     return z_h, a_h, z_out, a_out
    
    def predict(self, X):
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(a_out, axis=1)
        return y_pred
    
    def fit(self, X_train, y_train, X_test, y_test):
        
        assert X_train.ndim == 2, f"X_train must be 2D, got {X_train.shape}"
        assert y_train.ndim == 1, f"y_train must be 1D, got {y_train.shape}"
        
        if self.minibatch_size <= 0:
            self.minibatch_size = X_train.shape[0]
        # n_output = np.unique(y_train).shape[0]
        n_output = 11
        n_features = X_train.shape[1]
        
        
        scale_h = np.sqrt(2 / (n_features + self.n_hidden))
        scale_out = np.sqrt(2 / (self.n_hidden + n_output))
        self.b_h = np.zeros(self.n_hidden)
        self.b_out = np.zeros(n_output)
        self.w_h = self.random.normal(loc=0.0, scale=scale_h,size=(n_features, self.n_hidden))
        self.w_out = self.random.normal(loc=0.0,scale=scale_out,size=(self.n_hidden, n_output))
        
        y_train_enc = self._onehot(y_train, n_output)
        
        
        
        for epoch in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
                
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1,self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                X_batch = X_train[batch_idx]
                y_batch_enc = y_train_enc[batch_idx]
                z_h, a_h, z_out, a_out = self._forward(X_batch)
                
                delta_out = a_out - y_batch_enc
                sigmoid_der_h = a_h * (1. - a_h)
                delta_h = (np.dot(delta_out, self.w_out.T) * sigmoid_der_h)
                
                
                self.w_out -= self.eta * (np.dot(a_h.T, delta_out) + self.l2*self.w_out)
                self.b_out -= self.eta * np.sum(delta_out, axis=0)
                self.w_h -= self.eta * (np.dot(X_batch.T, delta_h) + self.l2*self.w_h)
                self.b_h -= self.eta * np.sum(delta_h, axis=0)
                
                
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_tr=y_train_enc,
                                        output=a_out)
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)
            train_acc = np.mean(y_train == y_train_pred)
            test_acc = np.mean(y_test == y_test_pred)
            self.cost.append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['test_acc'].append(test_acc)
            sys.stderr.write(f"\rEpoch {epoch+1}/{self.epochs} | Cost: {cost:.2f} | Train/Test Acc.: {train_acc*100:.2f}%/{test_acc*100:.2f}%")

            sys.stderr.flush()
            
            
        return self
