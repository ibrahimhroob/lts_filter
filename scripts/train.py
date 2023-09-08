"""
Author: Ibrahim
Date: March 2023
"""

import os
import time
import torch
import logging
import argparse
import datetime

import numpy as np
import torch.nn as nn

import random

import lts.models.pointnet as pn
import lts.models.transformer as pct

from tqdm import tqdm
from pathlib import Path

from lts.util.slices_loader import ScansDataLoader

from sklearn.metrics import r2_score as r2score

##########################################################################
def parse_args():
    parser = argparse.ArgumentParser('lts_filter')
    parser.add_argument('--model', type=str, default='pct', help='options: pn or pct')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')

    parser.add_argument('--epoch', default=80, type=int, help='Epoch to run [default: 80]')

    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--parallel', action='store_true', default=False, help='Use parallel GPUs if available for training')

    parser.add_argument('--seq', type=str, default='20220420', help='Dataset')
    parser.add_argument('--log_dir', type=str, default='train_0', help='Log path')

    return parser.parse_args()

##########################################################################
# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##########################################################################
logger = None

def create_log_dir(logdir):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('point_reg')
    experiment_dir.mkdir(exist_ok=True)
    if logdir is None or logdir == "None":
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(logdir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    return log_dir, experiment_dir, checkpoints_dir

def create_logger(log_dir, model):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)    
    
    # Prevent logger from printing to console
    logger.propagate = False

    return logger

##########################################################################
def init_weights(m):
    classname = m.__class__.__name__
    conv = classname.find('Conv') != -1
    linear = classname.find('Linear') != -1
    if conv or linear:
        torch.nn.init.xavier_normal_(m.weight.data) 

##########################################################################
def fit(points, target_gt, model, device):
    points = points.float().to(device)
    points = points.transpose(2, 1)
    target_gt = target_gt.float().to(device)

    # Predict
    target_pred = model(points)    #get the predicted labels from the network
    target_pred = target_pred.contiguous().view(-1,) # or torch.reshape(target_pred, (-1,)), this step is just to flatten the tensor into 1D 
    
    target_gt = target_gt.view(-1, 1)[:, 0] #flatten the gt values 

    # Calculate loss
    loss = torch.sqrt(((target_pred - target_gt) ** 2).mean()) 
    r2 = r2score(target_pred.cpu().data.numpy(), target_gt.cpu().data.numpy())

    return loss, r2


##########################################################################
def train(args):
    def log_string(str):
        logger.info(str)
        print(str)    

    '''CREATE DIR'''
    log_dir, experiment_dir, checkpoints_dir = create_log_dir(args.log_dir)

    '''LOG'''
    logger = create_logger(log_dir, args.model)
    log_string('PARAMETER ...')
    log_string(args)

    ######################################################
    '''Data loader'''
    DATASET_PATH = os.path.join(os.environ["DATA"], 'sequence', args.seq, 'scans')
    log_string('DATASET_PATH %s' % DATASET_PATH)
    
    lidar_frames = os.listdir(DATASET_PATH)

    set_seed(42)
    random.shuffle(lidar_frames)

    #split the frames into train and val a good ratio for this would be 80% and 20%
    train, validate = np.split(lidar_frames, [int(0.8*len(lidar_frames))])  

    ######################################################
    # 1. Load data
    print("start loading training data ...")
    TRAIN_DATASET = ScansDataLoader(DATASET_PATH, train) 

    print("start loading validating data ...")
    VALIDATE_DATASET = ScansDataLoader(DATASET_PATH, validate)      
    
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
      
    validateDataLoader = torch.utils.data.DataLoader(VALIDATE_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=12,
                                                 pin_memory=True, drop_last=True)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of validating data is: %d" % len(VALIDATE_DATASET))

    ######################################################
    # 2. Define model
    model = pn
    if args.model == 'pct':
        model = pct

    regressor = model.get_model()

    '''DEVICE TYPE'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.device_count() > 1 and args.parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        regressor = nn.DataParallel(regressor)

    regressor.to(device)

    ######################################################
    '''Load model if exist, else init new model and start from scratch'''
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        regressor.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pre-trained model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        regressor = regressor.apply(init_weights)

    ######################################################
    '''Define the optimizer'''
    optimizer = torch.optim.AdamW(
        regressor.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    global_epoch = 0
    best_eval_loss = None

    model_parameters = filter(lambda p: p.requires_grad, regressor.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    log_string('model num of params: %d' % params)

    ######################################################
    # 3. Training and testing
    for epoch in range(start_epoch, args.epoch):
        '''Train on LiDAR slices'''
        log_string('\n\n**** Epoch %d (%d/%s) ****' % ( global_epoch + 1, epoch + 1, args.epoch))

        regressor = regressor.train()

        loss_sum, r2_sum = 0, 0
        
        for i, (points, target_gt, __) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0):
            loss, r2 = fit(points, target_gt, regressor, device)

            # Back-propagation
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()

            #accumulate the loss 
            loss_sum += loss
            r2_sum += r2

        train_loss = loss_sum / float(len(trainDataLoader))
        train_r2 = r2_sum / float(len(trainDataLoader))
                
        log_string('Training RMSE loss: %f' % (train_loss))
        log_string('Training r2: %f' % (train_r2))
        
        if epoch % 5 == 0:
            log_string('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')


        '''Evaluate on LiDAR slices'''
        with torch.no_grad():
            loss_sum, r2_sum = 0, 0

            regressor = regressor.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target_gt, scan_id) in tqdm(enumerate(validateDataLoader), total=len(validateDataLoader), smoothing=0):
                loss, r2 = fit(points, target_gt, regressor, device)

                loss_sum += loss
                r2_sum += r2

            eval_mean_loss = (loss_sum / float(len(validateDataLoader)))
            eval_mean_r2 = (r2_sum / float(len(validateDataLoader)))

            log_string('eval RMSE loss: %f' % (eval_mean_loss))
            log_string('eval r2: %f ' % ( eval_mean_r2))

            if(global_epoch == 0):
                best_eval_loss = eval_mean_loss    #at first run set the best_eval_loss to the first values of eval_mean_loss 

            if eval_mean_loss < best_eval_loss:
                best_eval_loss = eval_mean_loss
                log_string('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % (savepath))
                state = {
                    'epoch': epoch,
                    'model_state_dict': regressor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best eval RMSE score: *%f*' % ( best_eval_loss))
        global_epoch += 1

##########################################################################
if __name__ == '__main__':
    args = parse_args()
    train(args)
