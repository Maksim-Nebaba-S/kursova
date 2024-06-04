import os
from keras.callbacks import Callback
import tensorflow as tf 
from folder_manager import fresult
import numpy as np
from keras import backend as K

class CustomModelCheckpoint(Callback):
    def __init__(self, period, directory=f'weights/{fresult}/model_checkpoints'):
        super(CustomModelCheckpoint, self).__init__()
        self.period = period
        self.directory = directory
        # Ensure the directory exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            # Save the model
            model_path = os.path.join(self.directory, f'model_epoch_{epoch + 1}.h5')
            self.model.save(model_path)
            print(f'\nModel checkpoint saved at {model_path}')

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
class CyclicLR(Callback):
    def __init__(self, base_lr=1e-4, max_lr=1e-2, step_size=2000., mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.iterations = 0
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == 'triangular2':
            return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / float(2 ** (cycle - 1))
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))

    def on_train_begin(self, logs=None):
        logs = logs or {}
        if self.mode == 'exp_range':
            self.base_lr = logs.get('lr', self.base_lr)
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        self.iterations += 1
        lr = self.clr()
        K.set_value(self.model.optimizer.lr, lr)
        self.history.setdefault('lr', []).append(lr)
        self.history.setdefault('iterations', []).append(self.iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


    