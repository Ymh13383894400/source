# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets_yolo import get_train_model, yolo_body
from yolo_training import get_lr_scheduler
from callbacks import EvalCallback, LossHistory, ModelCheckpoint
from dataloader import YoloDatasets
from utils import get_anchors, get_classes, show_config
from utils_fit import fit_one_epoch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from yolo import YOLO
import six
import logging
from urllib.parse import urlparse
import cv2
import numpy as np
from tqdm import tqdm
import yolo



os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
s3_url = os.getenv("S3_ENDPOINT_URL", "http://s3.amazonaws.com")
if not (s3_url.startswith("http://") or s3_url.startswith("https://")):
    _url = f"https://{s3_url}"
s3_url = urlparse(s3_url)
s3_use_ssl = s3_url.scheme == 'https' if s3_url.scheme else True

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("SECRET_ACCESS_KEY", "")
os.environ["S3_ENDPOINT"] = s3_url.netloc
os.environ["S3_USE_HTTPS"] = "1" if s3_use_ssl else "0"
LOG = logging.getLogger(__name__)
# flags = tf.flags.FLAGS
flags = tf.compat.v1.flags

def preprocess(image, input_shape):
    """Preprocess functions in edge model inference"""
    # resize image with unchanged aspect ratio using padding by opencv
    h, w, _ = image.shape
    input_h, input_w = input_shape
    scale = min(float(input_w) / float(w), float(input_h) / float(h))
    nw = int(w * scale)
    nh = int(h * scale)
    image = cv2.resize(image, (nw, nh))
    new_image = np.zeros((input_h, input_w, 3), np.float32)
    new_image.fill(128)
    bh, bw, _ = new_image.shape
    new_image[
        int((bh - nh) / 2):(nh + int((bh - nh) / 2)),
        int((bw - nw) / 2):(nw + int((bw - nw) / 2)), :
    ] = image
    new_image /= 255.
    new_image = np.expand_dims(new_image, 0)  # Add batch dimension.
    return new_image


def create_input_feed(sess, new_image, img_data):
    """Create input feed for edge model inference"""
    input_feed = {}
    input_img_data = sess.graph.get_tensor_by_name('images:0')
    input_feed[input_img_data] = new_image
    input_img_shape = sess.graph.get_tensor_by_name('shapes:0')
    input_feed[input_img_shape] = [img_data.shape[0], img_data.shape[1]]
    return input_feed


def create_output_fetch(sess):
    """Create output fetch for edge model inference"""
    output_classes = sess.graph.get_tensor_by_name('output/classes:0')
    output_scores = sess.graph.get_tensor_by_name('output/scores:0')
    output_boxes = sess.graph.get_tensor_by_name('output/boxes:0')
    output_fetch = [output_classes, output_scores, output_boxes]
    return output_fetch


class Estimator:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        # sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.graph = tf.Graph()
        self.session = tf.compat.v1.Session(
            config=sess_config, graph=self.graph)

    def train(self, train_data, valid_data=None, **kwargs):
        """
        train
        """
        eager = False

        train_gpu = [0, ]

        classes_path = '../model_data/voc_classes.txt'

        anchors_path = '../model_data/yolo_anchors.txt'
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        model_path = '../model_data/yolo_weights.h5'
        # model_path = 'logs/best_epoch_weights.h5'
        # ------------------------------------------------------#
        #   input_shape     输入的shape大小，一定要是32的倍数
        # ------------------------------------------------------#
        input_shape = [416, 416]

        Init_Epoch = 0
        Freeze_Epoch = 50
        Freeze_batch_size = 16

        UnFreeze_Epoch = 300
        Unfreeze_batch_size = 8

        Freeze_Train = True

        Init_lr = 1e-2
        Min_lr = Init_lr * 0.01

        optimizer_type = "sgd"
        momentum = 0.937
        weight_decay = 5e-4

        lr_decay_type = 'cos'

        save_period = 10
        # ------------------------------------------------------------------#
        #   save_dir        权值与日志文件保存的文件夹
        # ------------------------------------------------------------------#
        save_dir = 'logs'

        eval_flag = True
        eval_period = 10

        num_workers = 1

        # train_annotation_path   训练图片路径和标签
        # val_annotation_path     验证图片路径和标签
        train_annotation_path = '2007_train.txt'
        val_annotation_path = '2007_val.txt'

        # 设置用到的显卡
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
        ngpus_per_node = len(train_gpu)

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        if ngpus_per_node > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = None
        print('Number of devices: {}'.format(ngpus_per_node))

        # 获取classes和anchor
        class_names, num_classes = get_classes(classes_path)
        anchors, num_anchors = get_anchors(anchors_path)

        # 判断是否多GPU载入模型和预训练权重
        if ngpus_per_node > 1:
            with strategy.scope():
                # 创建yolo模型
                model_body = yolo_body((None, None, 3), anchors_mask, num_classes, weight_decay)
                if model_path != '':
                    # 载入预训练权重
                    print('Load weights {}.'.format(model_path))
                    model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
                if not eager:
                    model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)
        else:
            # 创建yolo模型
            model_body = yolo_body((None, None, 3), anchors_mask, num_classes, weight_decay)
            if model_path != '':
                # 载入预训练权重
                print('Load weights {}.'.format(model_path))
                model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
            if not eager:
                model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)

        # 读取数据集对应的txt
        with open(train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(val_annotation_path, encoding='utf-8') as f:
            val_lines = f.readlines()
        num_train = len(train_lines)
        num_val = len(val_lines)

        show_config(
            classes_path=classes_path, anchors_path=anchors_path, anchors_mask=anchors_mask, model_path=model_path,
            input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
            num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
            total_step, wanted_step, wanted_epoch))

        if True:
            if Freeze_Train:
                freeze_layers = 184
                for i in range(freeze_layers): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

            # -------------------------------------------------------------------#
            #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
            # -------------------------------------------------------------------#
            batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 64
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            # ---------------------------------------#
            #   获得学习率下降的公式
            # ---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

            train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask,
                                            train=True)
            val_dataloader = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask,
                                          train=False)

            optimizer = {
                'adam': Adam(lr=Init_lr, beta_1=momentum),
                'sgd': SGD(lr=Init_lr, momentum=momentum, nesterov=True)
            }[optimizer_type]

            if eager:
                start_epoch = Init_Epoch
                end_epoch = UnFreeze_Epoch
                UnFreeze_flag = False

                gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate),
                                                     (tf.float32, tf.float32, tf.float32, tf.float32))
                gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate),
                                                         (tf.float32, tf.float32, tf.float32, tf.float32))

                gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
                gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

                if ngpus_per_node > 1:
                    gen = strategy.experimental_distribute_dataset(gen)
                    gen_val = strategy.experimental_distribute_dataset(gen_val)

                time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
                log_dir = os.path.join(save_dir, "loss_" + str(time_str))
                loss_history = LossHistory(log_dir)
                eval_callback = EvalCallback(model_body, input_shape, anchors, anchors_mask, class_names, num_classes,
                                             val_lines, log_dir, \
                                             eval_flag=eval_flag, period=eval_period)
                # 开始模型训练
                for epoch in range(start_epoch, end_epoch):
                    # 如果模型有冻结学习部分
                    # 则解冻，并设置参数
                    if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                        batch_size = Unfreeze_batch_size

                        # 判断当前batch_size，自适应调整学习率
                        nbs = 64
                        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                        # 获得学习率下降的公式
                        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                        for i in range(len(model_body.layers)):
                            model_body.layers[i].trainable = True

                        epoch_step = num_train // batch_size
                        epoch_step_val = num_val // batch_size

                        if epoch_step == 0 or epoch_step_val == 0:
                            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                        train_dataloader.batch_size = batch_size
                        val_dataloader.batch_size = batch_size

                        gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate),
                                                             (tf.float32, tf.float32, tf.float32, tf.float32))
                        gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate),
                                                                 (tf.float32, tf.float32, tf.float32, tf.float32))

                        gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
                        gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

                        if ngpus_per_node > 1:
                            gen = strategy.experimental_distribute_dataset(gen)
                            gen_val = strategy.experimental_distribute_dataset(gen_val)

                        UnFreeze_flag = True

                    lr = lr_scheduler_func(epoch)
                    K.set_value(optimizer.lr, lr)

                    fit_one_epoch(model_body, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                                  gen, gen_val,
                                  end_epoch, input_shape, anchors, anchors_mask, num_classes, save_period, save_dir,
                                  strategy)

                    train_dataloader.on_epoch_end()
                    val_dataloader.on_epoch_end()
            else:
                start_epoch = Init_Epoch
                end_epoch = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch

                if ngpus_per_node > 1:
                    with strategy.scope():
                        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                else:
                    model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                # -------------------------------------------------------------------------------#
                #   训练参数的设置
                #   logging         用于设置tensorboard的保存地址
                #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
                #   lr_scheduler       用于设置学习率下降的方式
                #   early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
                # -------------------------------------------------------------------------------#
                time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
                log_dir = os.path.join(save_dir, "loss_" + str(time_str))
                logging = TensorBoard(log_dir)
                loss_history = LossHistory(log_dir)
                checkpoint = ModelCheckpoint(
                    os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=save_period)
                checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"),
                                                  monitor='val_loss', save_weights_only=True, save_best_only=False,
                                                  period=1)
                checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"),
                                                  monitor='val_loss', save_weights_only=True, save_best_only=True,
                                                  period=1)
                early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
                lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
                eval_callback = EvalCallback(model_body, input_shape, anchors, anchors_mask, class_names, num_classes,
                                             val_lines, log_dir, \
                                             eval_flag=eval_flag, period=eval_period)
                callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler,
                             eval_callback]

                if start_epoch < end_epoch:
                    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                               batch_size))
                    model.fit(
                        x=train_dataloader,
                        steps_per_epoch=epoch_step,
                        validation_data=val_dataloader,
                        validation_steps=epoch_step_val,
                        epochs=end_epoch,
                        initial_epoch=start_epoch,
                        use_multiprocessing=True if num_workers > 1 else False,
                        workers=num_workers,
                        callbacks=callbacks
                    )
                # 如果模型有冻结学习部分
                # 则解冻，并设置参数
                if Freeze_Train:
                    batch_size = Unfreeze_batch_size
                    start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                    end_epoch = UnFreeze_Epoch

                    # 判断当前batch_size，自适应调整学习率
                    nbs = 64
                    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                    # 获得学习率下降的公式
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
                    callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler,
                                 eval_callback]

                    for i in range(len(model_body.layers)):
                        model_body.layers[i].trainable = True
                    if ngpus_per_node > 1:
                        with strategy.scope():
                            model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                    else:
                        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

                    epoch_step = num_train // batch_size
                    epoch_step_val = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                    train_dataloader.batch_size = Unfreeze_batch_size
                    val_dataloader.batch_size = Unfreeze_batch_size

                    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                               batch_size))
                    model.fit(
                        x=train_dataloader,
                        steps_per_epoch=epoch_step,
                        validation_data=val_dataloader,
                        validation_steps=epoch_step_val,
                        epochs=end_epoch,
                        initial_epoch=start_epoch,
                        use_multiprocessing=True if num_workers > 1 else False,
                        workers=num_workers,
                        callbacks=callbacks
                    )
            # return {"loss": float(np.mean(loss))}

    def evaluate(
            self,
            valid_data,
            model_path="",
            class_names="",
            input_shape=(
                    352,
                    640),
            **kwargs):
        """
        validate
        """
        # precision, recall, all_precisions, all_recalls = (
        #     validate(model_path=model_path,
        #              test_dataset=valid_data.x,
        #              class_names=class_names,
        #              input_shape=input_shape)
        # )
        # return {
        #     "recall": recall, "precision": precision
        # }

    def avg_checkpoints(self):
        """
        Average the last N checkpoints in the model_dir.
        """

        LOG.info("average checkpoints start .......")

        # with self.session.as_default() as sess:
        #
        #     yolo_config = YoloConfig()
        #     model = Yolo3(sess, False, yolo_config)
        #
        #     model_dir = model.model_dir
        #     num_last_checkpoints = 5
        #     global_step = model.global_step.eval()
        #     global_step_name = model.global_step.name.split(":")[0]
        #
        #     checkpoint_state = tf.train.get_checkpoint_state(model_dir)
        #     if not checkpoint_state:
        #         logging.info(
        #             "# No checkpoint file found in directory: %s" %
        #             model_dir)
        #         return None
        #
        #     # Checkpoints are ordered from oldest to newest.
        #     checkpoints = (
        #         checkpoint_state.all_model_checkpoint_paths[
        #                         - num_last_checkpoints:]
        #     )
        #
        #     if len(checkpoints) < num_last_checkpoints:
        #         logging.info(
        #             "# Skipping averaging checkpoints because"
        #             " not enough checkpoints is avaliable.")
        #         return None
        #
        #     avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
        #     if not tf.gfile.Exists(avg_model_dir):
        #         logging.info(
        #             "# Creating new directory %s "
        #             "for saving averaged checkpoints." %
        #             avg_model_dir)
        #         tf.gfile.MakeDirs(avg_model_dir)
        #
        #     logging.info("# Reading and averaging "
        #                  "variables in checkpoints:")
        #     var_list = tf.contrib.framework.list_variables(checkpoints[0])
        #     var_values, var_dtypes = {}, {}
        #     for (name, shape) in var_list:
        #         if name != global_step_name:
        #             var_values[name] = np.zeros(shape)
        #
        #     for checkpoint in checkpoints:
        #         logging.info("        %s" % checkpoint)
        #         reader = tf.contrib.framework.load_checkpoint(checkpoint)
        #         for name in var_values:
        #             tensor = reader.get_tensor(name)
        #             var_dtypes[name] = tensor.dtype
        #             var_values[name] += tensor
        #
        #     for name in var_values:
        #         var_values[name] /= len(checkpoints)
        #
        #     # Build a graph with same variables in
        #     # the checkpoints, and save the averaged
        #     # variables into the avg_model_dir.
        #     with tf.Graph().as_default():
        #         tf_vars = [
        #             tf.get_variable(
        #                 v,
        #                 shape=var_values[v].shape,
        #                 dtype=var_dtypes[name]) for v in var_values]
        #
        #         placeholders = [
        #             tf.placeholder(
        #                 v.dtype,
        #                 shape=v.shape) for v in tf_vars]
        #         assign_ops = [
        #             tf.assign(
        #                 v,
        #                 p) for (
        #                 v,
        #                 p) in zip(
        #                 tf_vars,
        #                 placeholders)]
        #         global_step_var = tf.Variable(
        #             global_step, name=global_step_name, trainable=False)
        #         saver = tf.train.Saver(tf.global_variables())
        #
        #         with tf.Session() as sess:
        #             sess.run(tf.global_variables_initializer())
        #             for p, assign_op, (name, value) in zip(
        #                     placeholders, assign_ops,
        #                     six.iteritems(var_values)):
        #                 sess.run(assign_op, {p: value})
        #
        #             # Use the built saver to save the averaged checkpoint.
        #             # Only keep 1 checkpoint and the best checkpoint will
        #             # be moved to avg_best_metric_dir.
        #             saver.save(
        #                 sess, os.path.join(
        #                     avg_model_dir, "translate.ckpt"))

        logging.info("average checkpoints end .......")

    def predict(self, data, input_shape=None, **kwargs):
        # img_data_np = np.array(data)
        # with self.session.as_default():
        #     new_image = preprocess(img_data_np, input_shape)
        #     input_feed = create_input_feed(
        #         self.session, new_image, img_data_np)
        #     output_fetch = create_output_fetch(self.session)
        #     output = self.session.run(output_fetch, input_feed)
        #
        #     return output
        return yolo.detect_image(data)

    def load(self, model_url):
        with self.session.as_default():
            with self.session.graph.as_default():
                with tf.gfile.FastGFile(model_url, 'rb') as handle:
                    LOG.info(f"Load model {model_url}, "
                             f"ParseFromString start .......")
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(handle.read())
                    LOG.info("ParseFromString end .......")

                    tf.import_graph_def(graph_def, name='')
                    LOG.info("Import_graph_def end .......")
        LOG.info("Import model from pb end .......")

    def save(self, model_path=None):
        """
        save model as a single pb file from checkpoint
        """
        # model_dir = ""
        # model_name = "model.pb"
        # if model_path:
        #     model_dir, model_name = os.path.split(model_path)
        # logging.info("save model as .pb start .......")
        # tf.reset_default_graph()
        #
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        #
        # with tf.Session(config=config) as sess:
        #
        #     yolo_config = YoloConfig()
        #
        #     model = Yolo3(sess, False, yolo_config)
        #     if not (model_dir and os.path.isdir(model_dir)):
        #         model_dir = model.model_dir
        #     input_graph_def = sess.graph.as_graph_def()
        #     output_tensors = [model.boxes, model.scores, model.classes]
        #     output_tensors = [t.op.name for t in output_tensors]
        #     graph = tf.graph_util.convert_variables_to_constants(
        #         sess, input_graph_def, output_tensors)
        #     tf.train.write_graph(graph, model_dir, model_name, False)

        logging.info("save model as .pb end .......")
