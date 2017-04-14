import os.path
import argparse
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model
from keras.engine.training import _make_batches
from keras import backend as K
from keras import losses
from keras import initializers
from keras.datasets import mnist

BLOCK_SIZE = 28
margs = {'batch_size': 100, 'original_dim': np.square(BLOCK_SIZE), 'latent_dim': 2, 'intermed_dim': 512, 'nepochs': 100, 'optimizer': 'adam'}

def get_decoder(model, latent_dim, batch_size=1):
    z = Input(batch_shape=(batch_size, latent_dim), name='z')
    h_decoded = model.get_layer('decoder_h')(z)
    x_decoded_mean = model.get_layer('decoder_mean')(h_decoded)
    mdl = Model(z, x_decoded_mean)
    return mdl

def get_model(batch_size, original_dim, latent_dim, intermed_dim, objective, optimizer):

    x = Input(batch_shape=(batch_size, original_dim), name='x')

    # build encoder
    h_layer = Dense(intermed_dim,
        activation='relu',
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
        name='h')
    z_mean_layer = Dense(latent_dim,
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
        name='z_mean')
    z_stddev_layer = Dense(latent_dim,
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
        activation='softplus',
        name='z_stddev')
    h = h_layer(x)
    z_mean = z_mean_layer(h)
    z_stddev = z_stddev_layer(h)

    # sample latents
    def sampling(args):
        z_mean, z_stddev = args
        eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + z_stddev*eps
    z = Lambda(sampling)([z_mean, z_stddev])

    # build decoder
    h_decoded_layer = Dense(intermed_dim,
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
        activation='relu',
        name='decoder_h')
    x_decoded_mean_layer = Dense(original_dim,
        activation='sigmoid',
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
        name='decoder_mean')
    h_decoded = h_decoded_layer(z)
    x_decoded_mean = x_decoded_mean_layer(h_decoded)

    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim*losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = 0.5*K.sum(z_stddev + K.square(z_mean) - 1 - K.log(z_stddev), axis=-1)
        return kl_loss + xent_loss

    model = Model(x, x_decoded_mean)
    model.compile(optimizer=optimizer,
        loss=vae_loss,
        metrics=[losses.binary_crossentropy])

    z_args = concatenate([z_mean, z_stddev], axis=-1)
    enc_model = Model(x, z_args)
    # enc_model = Model(x, [z_mean, z_stddev])
    if objective == 'sgvb':
        return model, enc_model
    # 1/0

    def wake_loss(x, x_decoded_mean):
        """
        xent_loss = binary cross-entropy
        w = -log N(z; 0, I)
        """
        return losses.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = original_dim*losses.binary_crossentropy(x, x_decoded_mean)
        # w = 0.5*K.sum(K.square(z), axis=-1)
        # return xent_loss + w
    
    # z0, sampled from z_mean, z_stddev is now Input!
    # z0 = Input(batch_shape=(batch_size, latent_dim), name='z0')
    # h_decoded0 = h_decoded_layer(z0)
    # x_decoded_mean0 = x_decoded_mean_layer(h_decoded0)
    
    wake_model = Model(x, x_decoded_mean)
    h_layer.trainable = False
    z_mean_layer.trainable = False
    z_stddev_layer.trainable = False
    wake_model.compile(optimizer=optimizer, loss=wake_loss)
    
    def sleep_loss(z_true, z_args):
        """
        -log N(z_true; z_mean, I*z_stddev^2)
        """
        z_mean = z_args[:,:latent_dim]
        z_stddev = z_args[:,latent_dim:]
        # log(det(diag(z_stddev^2))) = sum(log(z_stddev^2))
        logdet = 2*K.sum(K.log(z_stddev), axis=-1)
        z0 = K.square(z_mean - z_true)
        w = K.sum(z0*(1/K.square(z_stddev)), axis=-1)
        return 0.5*(logdet + w)

    z_args = concatenate([z_mean, z_stddev], axis=-1)
    sleep_model = Model(x, z_args)
    sleep_model.compile(optimizer=optimizer, loss=sleep_loss)

    return wake_model, sleep_model

def sample_z(args, nsamps=1):
    z_mean, z_stddev = args
    # eps = np.random.randn(*((nsamps, z_mean.flatten().shape[0])))
    eps = np.random.randn(nsamps, z_mean.shape[0], z_mean.shape[1])
    return z_mean + z_stddev*eps

def train_wakesleep(X, wake_model, sleep_model, dec_model, latent_dim, nepochs, batch_size, model_file):
    n = len(X)
    index_array = np.arange(n)
    batches = _make_batches(n, batch_size)
    stat_index = np.round(0.1*len(batches))

    z_base_mean = np.zeros([batch_size, latent_dim])
    z_base_sdev = np.ones([batch_size, latent_dim])
    z_base_params = (z_base_mean, z_base_sdev)
    for i in xrange(nepochs):
        np.random.shuffle(index_array)
        print 'Epoch {}/{}'.format(i, nepochs)
        s_losses = []
        w_losses = []
        if i % 10 == 0 or i == nepochs-1:
            wake_model.save_weights(model_file.format(epoch=i))
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            if batch_index == 0:
                print '[',
            if batch_index % stat_index == 0:
                print '=',
            batch_ids = index_array[batch_start:batch_end]
            x_batch = X[batch_ids]

            # wake phase
            w_loss = wake_model.train_on_batch(x_batch, x_batch)

            # sleep phase
            z_batch_sleep = sample_z(z_base_params)[0]
            x_batch_sleep = dec_model.predict(z_batch_sleep, batch_size=batch_size)
            s_loss = sleep_model.train_on_batch(x_batch_sleep, z_batch_sleep)

            w_losses.append(w_loss)
            s_losses.append(s_loss)
        print '] w_loss: {}, s_loss: {}'.format(np.array(w_losses).mean(), np.array(s_losses).mean())

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # convert data from [n x 28 x 28] -> [n x 784], between 0.0 and 1.0
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, y_train, x_test, y_test

def make_modelfile(model_dir, run_name):
    return os.path.join(model_dir, run_name + '-{epoch:02d}.h5')

def get_callbacks(run_name, model_dir, log_dir):
    # prepare to save model checkpoints
    chkpt_filename = make_modelfile(model_dir, run_name)
    checkpt = ModelCheckpoint(chkpt_filename, monitor='val_loss', save_best_only=True, save_weights_only=True, period=1)
    logs = TensorBoard(log_dir=os.path.join(log_dir, run_name))
    callbacks = [logs, checkpt]
    return callbacks

def load_models(model_file, batch_size_x=1, batch_size_z=1):
    objective = 'sgvb' if 'sgvb' in model_file else 'ws'
    model, enc_model = get_model(batch_size_x, margs['original_dim'], margs['latent_dim'], margs['intermed_dim'], objective, margs['optimizer'])
    dec_model = get_decoder(model, margs['latent_dim'], batch_size=batch_size_z)
    model.load_weights(model_file)
    return model, enc_model, dec_model

def save_model(model, run_name, model_dir):
    # save model structure
    outfile = os.path.join(model_dir, run_name + '.yaml')
    with open(outfile, 'w') as f:
        f.write(model.to_yaml())

def main(run_name, objective='sgvb', model_dir='models', log_dir='logs',
    batch_size=margs['batch_size'],
    original_dim=margs['original_dim'],
    latent_dim=margs['latent_dim'],
    intermed_dim=margs['intermed_dim'],
    nepochs=margs['nepochs'], optimizer=margs['optimizer']):

    run_name += '-' + objective
    model, enc_model = get_model(batch_size, original_dim, latent_dim, intermed_dim, objective, optimizer)
    save_model(model, run_name, model_dir)
    callbacks = get_callbacks(run_name, model_dir, log_dir)

    x_train, y_train, x_test, y_test = load_data()
    if objective == 'sgvb':
        model.fit(x_train, x_train,
            shuffle=True,
            epochs=nepochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(x_test, x_test))
    else:
        dec_model = get_decoder(model, latent_dim, batch_size=batch_size)
        train_wakesleep(x_train, model, enc_model, dec_model, latent_dim, nepochs, batch_size, make_modelfile(model_dir, run_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('-o', '--objective', required=True,
        choices=['sgvb', 'ws'],
        help='either sgvb or ws')
    args = parser.parse_args()
    main(run_name=args.run_name, objective=args.objective)
