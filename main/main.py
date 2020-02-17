import os
import random
import datetime
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorboard.plugins.hparams import api as hp

import CSR_Net
import util

# confirm tf is using GPU
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_device$
# input('Press enter of gpu settings are good')

# gundersena@75.86.178.105:~/Desktop/crimata-super-res/train/logs/weights ~/Desktop
# scp rm -r gundersena.75.86.178.105:~/Desktop/crimata-super-res/main
# scp -r ~/Desktop/crimata-super-res/main gundersena@75.86.178.105:~/Desktop/crimata-super-res


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def make_parser():
    """creates argument parser from train and eval"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands')

    # train
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)

    train_parser.add_argument('-i','--model-id')
    train_parser.add_argument('-c','--from_ckpt')
    train_parser.add_argument('-k','--new-data')
    train_parser.add_argument('-d','--dim-size',type=int)
    train_parser.add_argument('-x','--num-files',type=int)
    # train_parser.add_argument('-t','--train-file')
    # train_parser.add_argument('-v','--val-file')
    train_parser.add_argument('-e','--epochs',type=int)
    train_parser.add_argument('-b','--batch-size',type=int)
    train_parser.add_argument('-o','--cycle-length',type=int)
    train_parser.add_argument('-m','--max-lr',type=float)
    train_parser.add_argument('-n','--min-lr',type=float)

    # eval
    eval_parser = subparsers.add_parser('eval')
    eval_parser.set_defaults(func=eval)

    eval_parser.add_argument('-i','--model-id')
    eval_parser.add_argument('-n','--num-examples',type=int)
    eval_parser.add_argument('-w','--wavfile-list')
    eval_parser.add_argument('-r','--scale',type=int)
    eval_parser.add_argument('-s','--sample-rate',type=int)
    eval_parser.add_argument('-a','--make-audio')
    eval_parser.add_argument('-c','--from-ckpt', default='True')

    return parser


def train(args):
    """High-level method for training a model"""
    # load data
    x_train, y_train, n_sam = util.load_data(args, type='train', num_files=args.num_files, full_data=True)
    x_val, y_val = util.load_data(args, type='val', num_files=int(np.floor(args.num_files*0.3)))

    # callbacks
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=f'logs/weights/weights.{args.model_id}.tf',
        monitor='val_loss', save_best_only=True, save_weights_only=True, mode='auto')

    # smart_learn = CSR_Net.util.SGDRScheduler(min_lr=args.min_lr, max_lr=args.max_lr,
    #     steps_per_epoch=np.ceil(n_sam/args.batch_size), cycle_length=args.cycle_length)

    # lr_finder = CSR_Net.util.LRFinder(min_lr=1e-7, max_lr=3e-2,
    #     steps_per_epoch=np.ceil(n_sam/args.batch_size), epochs=args.epochs)

    # logdir = f'logs/fit/{args.model_id}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # hparams = {'max_lr':args.max_lr, 'min_lr':args.min_lr, 'cycle_length':args.cycle_length}
    # param_logger = hp.KerasCallback(logdir, hparams)
    #
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1,
    #     write_graph=True, update_freq='epoch')

    # make model
    model = make_model(args)

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.max_lr)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=args.max_lr, momentum=0.8, nesterov=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # final review
    util.review_model(args, x_train, y_train)

    # train model
    model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs,
        callbacks=[checkpointer],
        validation_data=[x_val, y_val], shuffle=True)

    # plot loss and lr metrics
    # lr_finder.plot_lr()
    # lr_finder.plot_loss()


def eval(args):
    """test the model on real audio"""
    # make model
    model = make_model(args)

    # create list of file names
    file_list = []
    with open(args.wavfile_list) as f:
        for line in f:
            file_list.append(line) # this is gonna get pretty big for a real dataset...

    # eval on random sample of files
    file_list = random.sample(file_list, args.num_examples)
    for idx, line in enumerate(file_list):
        file = line.rstrip('\n')
        CSR_Net.util.eval_wav(file, args, model)


def make_model(args):
    """define a graph and compile model"""
    model = CSR_Net.MlRes()

    if args.from_ckpt == 'True':
        model.load_weights((f'logs/weights/weights.{args.model_id}.tf'))

    return model


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
