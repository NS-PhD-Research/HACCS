# The model owner is considered as client
import os
import sys
import time
import asyncio
import argparse
import threading
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers import websocket_client
from syft.workers.websocket_client import WebsocketClientWorker

from concurrent import futures
import grpc
import logging

# Add common folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common')
sys.path.append(pwd)

# Add summary folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common', 'summary')
sys.path.append(pwd)

# Add models folder path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models')
sys.path.append(pwd)

# Add scheduler folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scheduler')
sys.path.append(pwd)

import devicetocentral_pb2
import devicetocentral_pb2_grpc

from utilities import Utility as util
from modelFactory import ModelFactory as mdlftry
from schedulerFactory import SchedulerFactory as schedftry

LOCK_TRACE = False
EXPECTED_EPOCH_DURATION = 10.0

def generateNetworkDelays(n=30):

    MODEL_SIZE_IN_BITS = float(8 * 10 * 1024 * 1024)   # 10 MB

    np.random.seed(0x12345678)
    delays = []

    # fast, medium, slow, very slow probabilities
    probs = np.array([0.6, 0.2, 0.14, 0.06])
    samples = np.random.multinomial(1, probs, n)

    for sample in samples:

        # Just generate a random latency in seconds (20-200ms)
        latency = np.random.uniform(0.02, 0.2)

        # BW values in megabits per sec
        draw = np.array([np.random.uniform(75.0, 100.0),
                         np.random.uniform(50.0,  75.0),
                         np.random.uniform(25.0,  50.0),
                         np.random.uniform( 1.0,  25.0)])

        bw_arr = np.multiply(sample, draw)
        bw_idx = np.nonzero(bw_arr)
        bw = float(bw_arr[bw_idx]) * 1024.0 * 1024.0

        nw_delay = latency + (MODEL_SIZE_IN_BITS / bw)
        delays.append(nw_delay)

    return delays

def generateDelays(n=30):

    global EXPECTED_EPOCH_DURATION

    np.random.seed(1111)
    delays = []

    nw_delays = generateNetworkDelays(n)

    # fast, medium, slow, very slow probabilities
    probs = np.array([0.6, 0.2, 0.14, 0.06])
    samples = np.random.multinomial(1, probs, n)

    for idx, sample in enumerate(samples):
        draw = np.array([np.random.uniform(0.0, 0.5),
                         np.random.uniform(0.5, 1.0),
                         np.random.uniform(1.0, 1.5),
                         np.random.uniform(1.5, 2.0)])

        delay_arr = np.multiply(sample, draw)
        delay_idx = np.nonzero(delay_arr)
        cpu_delay = float(delay_arr[delay_idx]) * float(EXPECTED_EPOCH_DURATION)
        delays.append(cpu_delay + nw_delays[idx])

    return delays

usage_array = generateDelays(110)
usage_iter = 0

id_count = 1
dev_drop_seed = 1

class DeviceToCentralServicer(devicetocentral_pb2_grpc.DeviceToCentralServicer):
    
    def __init__(self, args):
        super().__init__()
        self.available_devices = {}
        self.n_available_devices = 0
        self.n_device_summaries = 0
        self.lockObj = threading.Lock()
        self.scheduler = schedftry.getScheduler(args.scheduler)
        
    def lock(self, traceStr="default"):
        global LOCK_TRACE
        if LOCK_TRACE: print("Server Lock   - " + traceStr)
        self.lockObj.acquire()

    def unlock(self, traceStr=""):
        global LOCK_TRACE
        if LOCK_TRACE: print("Server Unlock - " + traceStr)
        self.lockObj.release()

    def RegisterToCentral(self, request, context):
        global id_count
        msg = request.ip + ':' + str(request.flport)
        # id = util.get_id(msg)
        
        self.lock()
        id_count += 1
        id = str(id_count)
        if id not in self.available_devices:
            self.available_devices[id] = {}
            self.available_devices[id]['id'] = id
            self.available_devices[id]['ip'] = request.ip
            self.available_devices[id]['flport'] = request.flport
            self.available_devices[id]['fail'] = 0
            self.n_available_devices += 1
        self.unlock()

        logging.info('Registered client ' + request.ip + ':' + str(request.flport) + '...')
        
        return devicetocentral_pb2.RegStatus(
            success = True,
            id = id
        )
    
    def HeartBeat(self, request, context):
        self.lock()
        self.available_devices[request.id]['ncpus'] = request.ncpus
        self.available_devices[request.id]['load'] = request.load15
        self.available_devices[request.id]['virtual_mem'] = request.virtual_mem
        self.available_devices[request.id]['battery'] = request.battery
        self.unlock()

        return devicetocentral_pb2.Pong(
            ack = True,
        )

    def SendSummary(self, request, context):

        global usage_array, usage_iter

        self.lock()
        usage = usage_array[usage_iter]
        self.available_devices[request.id]['cpu_usage'] = usage
        usage_iter += 1
        self.available_devices[request.id]['summary'] = request.summary
        self.available_devices[request.id]['summary_type'] = request.type
        self.n_device_summaries += 1
        if self.n_available_devices == self.n_device_summaries:
            self.scheduler.notify_worker_update(self.available_devices)        
        self.unlock()

        return devicetocentral_pb2.SummaryAck(
            ack = True,
        )

    def schedule_best_worker_instances(self, available_devices, client_threshold=10):
        self.lock()
        workers = self.scheduler.select_worker_instances(available_devices, client_threshold)
        self.unlock()
        return workers


 # Loss function

@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)

async def fit_model_on_worker(worker, traced_model, batch_size, max_nr_batches, lr, dataset, delay):
    """
    Send the model to the worker and fit the model on the worker's training data.

    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.

    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,           # only run (1 epoch * batch_size samples) for each round
        optimizer="SGD",
        optimizer_args={"lr": lr, "momentum": 0.9},
    )
    latency = delay

    train_config.send(worker)
    loss = await worker.async_fit(dataset_key=dataset + '_TRAIN', return_ids=[0])      # TODO: add deadline here
    model = train_config.model_ptr.get().obj
    
    fit_time = latency

    return worker.id, model, loss, fit_time

def evaluate_model_on_worker(model_identifier, worker, dataset_key, model, nr_bins, batch_size, device, print_target_hist=False):
    model.eval()

    # Create and send train config
    train_config = sy.TrainConfig(batch_size=batch_size, model=model, loss_fn=loss_fn, optimizer_args=None, epochs=1)

    train_config.send(worker)

    result = worker.evaluate(dataset_key=dataset_key, return_histograms=True, nr_bins=nr_bins, return_loss=True, return_raw_accuracy=True)
    test_loss = result["loss"]
    correct = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_pred = result["histogram_predictions"]
    hist_target = result["histogram_target"]

    if print_target_hist:
        print("Target histogram: ", hist_target)
    return(correct, len_dataset)

# set connection to all available devices. Useful for evaluating model on all devices
def set_worker_conn(hook, available_devices, available_instances, verbose):
    for devid in available_devices:
        if devid not in available_instances.keys():
            kwargs_websocket = {'host' : available_devices[devid]['ip'], 'hook' : hook, 'verbose' : verbose}
            clientWorker = WebsocketClientWorker(id = devid, port = available_devices[devid]['flport'], **kwargs_websocket)
            clientWorker.clear_objects_remote()
            available_instances[devid] = clientWorker

def per_epoch_device_drop(devcentral, n_device_to_drop):
    global dev_drop_seed
    random.seed(dev_drop_seed)
    dev_drop_seed += 1

    drop_device_keys = random.choices(list(devcentral.available_devices.keys()), k = n_device_to_drop)
    for key in drop_device_keys:
        devcentral.available_devices[key]['fail'] = 1

    return drop_device_keys

def per_epoch_recover_device(devcentral, drop_device_keys):
    for key in drop_device_keys:
        devcentral.available_devices[key]['fail'] = 0

async def train_and_eval(args, devcentral, client_threshold, verbose):
   
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    model = mdlftry.getModel(args.model).to(device)
    traced_model = torch.jit.trace(model, torch.zeros([1, 3, 32, 32], dtype=torch.float).to(device))
   
    max_fed_epoch = args.fedepoch
    learning_rate = args.lr
    dataset = args.dataset
    batch_size = 128

    while True:
        devcentral.lock("train block 1")
        if devcentral.n_available_devices < client_threshold:
            devcentral.unlock()
            time.sleep(3)
        else:
            devcentral.unlock()
            break
    
    # setup connection with all devices
    hook = sy.TorchHook(torch)
    available_instances = {}
    available_devices =  {}
    for curr_round in range(0, args.epochs):
        drop_device_keys = []

        while True:
            devcentral.lock("train block 2")
            if devcentral.n_available_devices == devcentral.n_device_summaries:
                # drop devices
                n_device_to_drop = int(devcentral.n_available_devices * args.drop/100.0)
                if n_device_to_drop != 0:
                    drop_device_keys = per_epoch_device_drop(devcentral, n_device_to_drop)
                    logging.info('EPOCH:' + str(curr_round) + ' Dropped devices id: ' +  ' '.join(drop_device_keys))
                
                list_available_devices = [copy.deepcopy(devcentral.available_devices[k]) for k in devcentral.available_devices if devcentral.available_devices[k]['fail'] == 0]

                # Recover devices
                if n_device_to_drop != 0:
                    per_epoch_recover_device(devcentral, drop_device_keys)

                # available devices for this epoch
                available_devices = {}
                for dev in list_available_devices:
                    available_devices[dev['id']] = dev.copy()

                devcentral.unlock()
                time.sleep(3)
                break
            else:
                devcentral.unlock()

        set_worker_conn(hook, available_devices, available_instances, verbose)
        schedule_start_time = time.time()
        selected_worker_instances = devcentral.schedule_best_worker_instances(available_devices, client_threshold)
        schedule_end_time = time.time()

        train_start_time = time.time()
        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=available_instances[devid],
                    traced_model=traced_model,
                    batch_size=batch_size,    # batch_size_list[worker.id],
                    max_nr_batches=max_fed_epoch,
                    lr=learning_rate,
                    dataset= dataset,
                    delay= available_devices[devid]['cpu_usage']
                )
                for devid in selected_worker_instances #if batch_size_list[worker.id]>0
            ]
        )

        train_end_time = time.time()
        actual_train_time = train_end_time - train_start_time

        models = {}
        loss_values = {}

        # Federate models (note that this will also change the model in models[0]
        max_latency = 0.0
        for worker_id, worker_model, worker_loss, latency in results:
            max_latency = max(max_latency, latency)
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        fit_time = actual_train_time + max_latency

        traced_model = utils.federated_avg(models)

        test_models=True    # eval after every round
        if test_models:
            # evaluate_model_locally(traced_model)
            _correct=0
            _total=0
            eval_start_time = time.time()
            for devid in available_instances:
                correct, total=evaluate_model_on_worker(        # test accuracy
                    model_identifier="Federated model",
                    worker=available_instances[devid],
                    dataset_key=dataset + "_TEST",
                    model=traced_model,
                    nr_bins=10,
                    batch_size=128,
                    device="cpu",
                    print_target_hist=False,
                )
                _correct+=correct
                _total+=total
                
                # For TiFL
                devcentral.available_devices[devid]["loss"] = 1.0 - (float(correct) / float(total))

                # For Oort
                global EXPECTED_EPOCH_DURATION
                expected_fit_time = EXPECTED_EPOCH_DURATION
                alpha = 2.0
                loss = 1.0 - (float(correct) / float(total))
                global_util = 1.0
                dev_latency = devcentral.available_devices[devid]["cpu_usage"]
                if expected_fit_time < dev_latency:
                    global_util = (expected_fit_time / fit_time) ** alpha
                devcentral.available_devices[devid]["util"] = loss * global_util

            eval_end_time = time.time()

            logging.info("EPOCH: " + str(curr_round) + " ACCURACY: "+ str(_correct/_total) + " FIT_TIME: " + str(fit_time))
            print("EPOCH:", curr_round, \
                  "AVG_ACCURACY:", round(_correct/_total, 8), \
                  "#WORKERS:",     len(selected_worker_instances), \
                  "SCHED TIME:",   round(schedule_end_time - schedule_start_time, 8), \
                  "TRAIN TIME:",   round(actual_train_time, 8), \
                  "TOTAL TRAIN:",  round(schedule_end_time - schedule_start_time + actual_train_time, 8), \
                  "FIT TIME:",     round(fit_time, 8), \
                  "EVAL TIME:",    round(eval_end_time - eval_start_time, 8))

        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

def grpcServe(devcentral):
    server = grpc.server(futures.ThreadPoolExecutor())
    devicetocentral_pb2_grpc.add_DeviceToCentralServicer_to_server(devcentral, server)
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    server.wait_for_termination()

def parse_arguments(args = sys.argv[1:]):

    #Parse arguments
    parser = argparse.ArgumentParser(description='Run FL using websocker client workers')

    parser.add_argument(
        '--epochs',
        type = int,
        default = 200,
        help = 'Number of epochs for FL',
    )

    parser.add_argument(
        '--fedepoch',
        type = int,
        default = 10,
        help = 'number of epochs on remote worker before fedavg',
    )

    parser.add_argument(
        '--dataset',
        type = str,
        default='CIFAR10',
        help = 'Dataset used',
    )

    parser.add_argument(
        '--model',
        type = str,
        default='LeNet',
        help = 'Model used',
    )

    parser.add_argument(
        '--threshold',
        type = int,
        default= 10,
        help = 'Max client threshold',
    )

    parser.add_argument(
        '--lr',
        type = float,
        default = 0.1,
        help = 'learning rate',
    )

    parser.add_argument(
        '--scheduler',
        type = str,
        default = 'PYSched',
        help = 'Scheduler type',
    )

    parser.add_argument(
        '--drop',
        type = float,
        default = 0.0,
        help = 'Device drop percentage',
    )

    parser.add_argument(
        '--cuda',
        action = 'store_true',
        help = 'use cuda if available',
    )

    parser.add_argument(
        '--seed',
        type = int, 
        default = 1,
        help = 'seed for torch random number generator',
    )

    parser.add_argument(
        '--savemodel',
        action = 'store_true',
        help = 'if set, save model'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='start websocket client worker in verbose mode: --verbose'
    )

    args = parser.parse_args(args = args)
    return args


if __name__ == '__main__':

    args = parse_arguments()

    # logging format
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    devcentral = DeviceToCentralServicer(args)
    
    #  grpc service for register, heartbeat
    grpcservice = threading.Thread(target=grpcServe, args=(devcentral, ))
    grpcservice.start()

    # train and eval models 
    client_threshold = args.threshold
    asyncio.get_event_loop().run_until_complete(
        train_and_eval(args, devcentral, client_threshold, args.verbose)
    )

    grpcservice.join()
    
