from datetime import datetime

import numpy as np
import tensorflow as tf

from tqdm import trange

class VAE:
    """Variational Auto Encoder(VAE)
    
    Returns
    -------
    object
        VAE model object
    """

    def __init__(self, n_in, n_hiddens, 
                batch_size=128, z_dim=10, seed=0):

        """initialize parameter of VAE
        
        Parameters
        ----------
        n_in : int
            input dimension
        n_hiddens : list
            number of hidden dims
        z_dim : int, optional
            middle layer dimention(feature dimention), by default 10
        seed : int, optional
            Random number generator's seed, by default 0
        """        

        self.n_in       = n_in
        self.n_hiddens  = n_hiddens
        self.z_dim = z_dim
        self.enc_weights = []
        self.enc_biases = []
        self.dec_weights = []
        self.dec_biases = []
        self.batch_size = batch_size

        self.x = None
        self.t = None
        self.sigma = None
        self.y = None
        self.z = None

        self.history   = {
            'train_loss' : [],
            'test_loss' : [],
            'train_recons' : [],
            'test_recons' : [],
            'train_kl' : [],
            'test_kl' : []
        }

        self.seed = seed
        self.rand_gen = np.random.RandomState(self.seed)
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

    def weight_variable(self, shape, name='w'):
        initial = self.rand_gen.randn(shape[0], shape[1]) / np.sqrt(shape[0]) * np.sqrt(2)
        return tf.get_variable(name, dtype=tf.float64, initializer=initial)

    def bias_variable(self, shape, name='b'):
        initial = np.zeros(shape)
        return tf.get_variable(name, dtype=tf.float64, initializer=initial)

    def inference(self, x):
        for i, n_hidden in enumerate(self.n_hiddens):
            if i == 0:
                input_x = x
                input_dim = self.n_in
            else:
                input_x = output
                input_dim = self.n_hiddens[i-1]
            self.enc_weights.append(self.weight_variable([input_dim, n_hidden], name='enc_w_' + str(i + 1)))
            self.enc_biases.append(self.bias_variable([n_hidden], name='enc_b_' + str(i + 1)))

            output = tf.nn.relu(tf.matmul(input_x, self.enc_weights[-1]) + self.enc_biases[-1])

        self.enc_weights.append(self.weight_variable([self.n_hiddens[-1], self.z_dim], name='enc_w_mu'))
        self.enc_biases.append(self.bias_variable([self.z_dim], name='enc_b_mu'))
        h_mu = tf.matmul(output, self.enc_weights[-1]) + self.enc_biases[-1]

        self.enc_weights.append(self.weight_variable([self.n_hiddens[-1], self.z_dim], name='enc_w_sigma'))
        self.enc_biases.append(self.bias_variable([self.z_dim], name='enc_b_sigma'))
        h_lnsigma = tf.matmul(output, self.enc_weights[-1]) + self.enc_biases[-1]

        z = h_mu + tf.random_normal([self.batch_size, self.z_dim], dtype=tf.float64) * tf.sqrt(tf.exp(h_lnsigma))

        for i, n_hidden in enumerate(self.n_hiddens[::-1]):
            if i == 0:
                input_x = z
                input_dim = self.z_dim
            else:
                input_x = output
                input_dim = self.n_hiddens[len(self.n_hiddens) - i]
            self.dec_weights.append(self.weight_variable([input_dim, n_hidden], name='dec_w_' + str(i + 1)))
            self.dec_biases.append(self.bias_variable([n_hidden], name='dec_b_' + str(i + 1)))

            output = tf.nn.relu(tf.matmul(input_x, self.dec_weights[-1]) + self.dec_biases[-1])
            

        self.dec_weights.append(self.weight_variable([self.n_hiddens[0], self.n_in], name='dec_w_last'))
        self.dec_biases.append(self.bias_variable([self.n_in], name='dec_b_last'))

        # y = tf.matmul(output, self.dec_weights[-1]) + self.dec_biases[-1]
        logits = tf.matmul(output, self.dec_weights[-1]) + self.dec_biases[-1]
        y = tf.nn.sigmoid(logits)

        return y, z, logits, h_mu, h_lnsigma

    def recons_loss(self, logits, t):
        recons_error = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=t,
                        logits=logits
                        ), axis=1))
        return recons_error
    
    def kl_loss(self, h_mu, h_lnsigma):
        return tf.reduce_mean(self.normal_kl(h_mu, h_lnsigma))

    def training(self, loss):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)
        return train_step

    def fit(self, x_train, x_test,
            epochs=500, verbose=1, 
            log_path='./log/log.txt', model_path='./model/model.ckpt'): 
        
        """fit model by given x
        
        Parameters
        ----------
        x_train : 2d-array
            Autoencoder train input and teacher data
        x_test : 2d-array
            Autoencoder test input data
        epochs : int, optional
            Number of epoch in learning, by default 500
        verbose : int, optional
            Output learning log or not, by default is 1, output log
        
        Returns
        -------
        dict
            Learning history including test loss and training loss
        """

        x = tf.placeholder(tf.float64, shape=[self.batch_size, self.n_in], name='x')
        t = tf.placeholder(tf.float64, shape=[self.batch_size, self.n_in], name='t')

        self.x = x
        self.t = t

        y, z, logits, h_mu, h_lnsigma = self.inference(x)
        recons_loss = self.recons_loss(logits, t)
        kl_loss = self.kl_loss(h_mu, h_lnsigma)
        loss = recons_loss + kl_loss
        train_step = self.training(loss)

        self.y = y
        self.z = z
        self.h_mu = h_mu

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run([init])

        self.sess = sess

        log = open(log_path, 'a')

        log.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        log.write(f'n_in: {self.n_in}')
        log.write(f'z_dim: {self.z_dim}')
        log.write('n_hidden :' + ','.join(map(str, self.n_hiddens)))

        n_train = len(x_train)
        n_batchs = n_train // self.batch_size

        for epoch in trange(epochs):
            self.rand_gen.shuffle(x_train)

            for i in range(n_batchs):
                start = i * self.batch_size
                end = start + self.batch_size
                
                sess.run([train_step], feed_dict={
                    x:x_train[start:end],
                    t:x_train[start:end]
                })

            train_loss = 0
            test_loss = 0
            train_recons = 0
            test_recons = 0
            train_kl = 0
            test_kl = 0

            for i in range(n_batchs):
                start = i * self.batch_size
                end = start + self.batch_size

                train_loss += loss.eval(session=sess, feed_dict={
                    x: x_train[start:end],
                    t: x_train[start:end]
                })

                train_recons += recons_loss.eval(session=sess, feed_dict={
                    x: x_train[start:end],
                    t: x_train[start:end]
                })

                train_kl += kl_loss.eval(session=sess, feed_dict={
                    x: x_train[start:end],
                    t: x_train[start:end]
                })

            n_test_batch = len(x_test) // self.batch_size
            for i in range(n_test_batch):
                start = i * self.batch_size
                end = start + self.batch_size
                test_loss += loss.eval(session=sess, feed_dict={
                    x: x_test[start:end],
                    t: x_test[start:end]
                })

                test_recons += recons_loss.eval(session=sess, feed_dict={
                    x: x_test[start:end],
                    t: x_test[start:end]
                })

                test_kl += kl_loss.eval(session=sess, feed_dict={
                    x: x_test[start:end],
                    t: x_test[start:end]
                })

            train_loss *= 1 / n_batchs
            test_loss *= 1 / n_test_batch
            train_kl *= 1 / n_batchs
            test_kl *= 1 / n_test_batch
            train_recons *= 1 / n_batchs
            test_recons *= 1 / n_test_batch

            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_recons'].append(train_recons)
            self.history['test_recons'].append(test_recons)
            self.history['train_kl'].append(train_kl)
            self.history['test_kl'].append(test_kl)

            log_str = f'epoch: {epoch} \
                    train loss: {train_loss} \
                    test loss: {test_loss}'
                   
            log.write(log_str + '\n')

            if verbose:
                print(f'epoch: {epoch} \
                    train loss: {train_loss} \
                    test loss: {test_loss}')

        saver.save(sess, model_path)
        log.close()

        return self.history

    def normal_kl(self, mu, lnsigma):
        """calculate kl divergence with standard gaussian distribution
        
        Parameters
        ----------
        mu : ndarray   
            mean tensor
        sigma : ndarray
            diagonal log root of covariance matrix
        Returns
        -------
        ndarray
            result of calculating KL-divergence
        """
        return 0.5 * tf.reduce_sum(tf.clip_by_value(tf.square(mu) + tf.exp(lnsigma) - lnsigma - 1.0, 1e-20, 1e+20) \
                                , reduction_indices=[1])

    def model_load(self, model_path):
        """load model with given model path
        
        Parameters
        ----------
        model_path : string
            model file path
        
        """

        tf.reset_default_graph()
        
        x = tf.placeholder(tf.float64, shape=[self.batch_size, self.n_in], name='x')
        t = tf.placeholder(tf.float64, shape=[self.batch_size, self.n_in], name='t')

        self.x = x
        self.t = t

        y, z, _, h_mu, _ = self.inference(x)

        self.y = y
        self.z = z
        self.h_mu = h_mu

        self.sess  = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

    def encode(self, x):
        """encode Z from given X
        
        Parameters
        ----------
        x : 2d-ndarray
            first layer input which want to encode
        
        Returns
        -------
        2d-ndarray
            corresponding hidden layer output Z
        """

        result_h_mu = []
        for i in range(x.shape[0] // self.batch_size):
            start = i * self.batch_size
            end = start + self.batch_size
            h_mu = self.h_mu.eval(session=self.sess, feed_dict={
                            self.x        : x[start:end]
                            })
            result_h_mu.append(h_mu)
        
        if (x.shape[0] % self.batch_size) != 0:
            start = x.shape[0] // self.batch_size * self.batch_size
            end = x.shape[0]
            rest_x = np.concatenate([x[start:end], np.zeros([self.batch_size - end + start, self.n_in])], axis=0)
            h_mu = self.h_mu.eval(session=self.sess, feed_dict={
                            self.x        : rest_x
                            })
            result_h_mu.append(h_mu[0:end - start])

        result_h_mu = np.concatenate(result_h_mu, axis=0)

        return result_h_mu

    def decode(self, z):
        """decode x from given Z
        
        Parameters
        ----------
        z : 2d-ndarray
            hidden layer input which want to decode
        
        Returns
        -------
        2d-ndarray
            corresponding last layer output
        """
        result_x = []
        for i in range(z.shape[0] // self.batch_size):
            start = i * self.batch_size
            end = start + self.batch_size
            x = self.y.eval(session=self.sess, feed_dict={
                            self.z        : z[start:end]
                            })
            result_x.append(x)
        
        if (z.shape[0] % self.batch_size) != 0:
            start = z.shape[0] // self.batch_size * self.batch_size
            end = z.shape[0]
            
            rest_z = np.concatenate([z[start:end], np.zeros([self.batch_size - end + start, self.z_dim])], axis=0)
            x = self.y.eval(session=self.sess, feed_dict={
                            self.z        : rest_z
                            })
            result_x.append(x[0:end - start])

        result_x = np.concatenate(result_x, axis=0)

        return result_x

    def generate(self, sample_num):
        """generate samples from given conditional inputs
        
        Parameters
        ----------
        sample_num : int
            generate sample num
        Returns
        -------
        2d-ndarray
            corresponding last layer outputs
        """

        result_x = []
        for i in range(sample_num // self.batch_size):
            z = self.rand_gen.normal(size=(self.batch_size, self.z_dim))
            x = self.y.eval(session=self.sess, feed_dict={
                        self.z : z
                            })
            result_x.append(x)
        
        if (sample_num % self.batch_size) != 0:
            z = self.rand_gen.normal(size=(self.batch_size, self.z_dim))
            x = self.y.eval(session=self.sess, feed_dict={
                        self.z        : z
                            })
            result_x.append(x[0:sample_num % self.batch_size])

        result_x = np.concatenate(result_x, axis=0)

        return result_x
