import numpy as np
import os
import itertools
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset.Dataset import TrainingDataset
from dataset.FruitDataset import FruitCountDataset
from models.MLNCount import MLNCountModel
from utils.util import get_config, find_local_maxima


def main():
    config = get_config('../configs/config.yaml')

    data_root = config['data_root']
    log_path = config['log_path']

    save_dir = os.path.join(log_path, config['exp_name'], 'Validation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seed = config['random_seed']
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # ------ load networks and set in evaluation mode
    model = MLNCountModel(opt=config, log_path=log_path)
    model.setup()
    model.initialize(opt=config)
    model.load_networks(which_epoch=config['which_epoch'])
    model.set_eval_mode()

    # ------ load dataset
    presence_absence_testset = FruitCountDataset(config=config, mode=TrainingDataset.VALIDATION,
                                                 name="PresenceAbsenceDataset_TEST", transforms=None,
                                                 data_root=data_root)

    testloader = DataLoader(presence_absence_testset, batch_size=1,
                            shuffle=False, num_workers=1, drop_last=False)

    # ------ computing maps
    maps = []
    labels = []

    for i, data in enumerate(testloader, 0):
        print("-- processing sample: ", i)
        inputs, label, _ = data
        model.set_input([inputs, label])
        output = model.inference_forward()

        map = output[0, :, :, :].cpu().detach().numpy()
        map = map.sum(axis=0)
        maps.append(map)
        labels.append(label)


    # ------ computing parameters for cross validation
    neighborhood_size = np.arange(config['min_r'], config['max_r'] + 1, config['step_r']).tolist()
    threshold = np.arange(config['min_thp'], config['max_thp'] + 1, config['step_thp']).tolist()

    param = [neighborhood_size, threshold]
    parameters = list(itertools.product(*param))

    RMSEs = []
    header = ['N', 'T', 'RMSE']
    data = pd.DataFrame(columns=header)

    for j in range(len(parameters)):

        print(f'\nSession {j} / {len(parameters) - 1}')
        print(f'Parameters: {parameters[j]}')

        absolute_errors = []
        for i in range(len(maps)):

            detection = find_local_maxima(maps[i], neighborhood_size=parameters[j][0], threshold=parameters[j][1])
            fruits = labels[i].cpu().detach().numpy()[0]
            absolute_errors.append(len(detection) - fruits)

        absolute_errors = np.array(absolute_errors)
        RMSE = np.sqrt(np.mean(np.square(absolute_errors), axis=0))

        RMSEs.append(RMSE)
        print(f'RMSE: {RMSE}')
        row = [parameters[j][0], parameters[j][1], RMSE[0]]
        data.loc[len(data)] = row

    min_RMSE = min(RMSEs)
    index = RMSEs.index(min(RMSEs))
    print(f'\nBest RMSE found in Session: {index}')
    print(f'RMSE: {min_RMSE}')
    print(f'Parameters: {parameters[index]}')

    data.to_csv(path_or_buf=os.path.join(save_dir, 'RMSE_val.csv'), index=False, header=header, sep=',')

    best_data = data.loc[data['N'] == parameters[index][0]]
    best_data.to_csv(path_or_buf=os.path.join(save_dir, 'best_RMSE_val.csv'), index=False, header=header, sep=',')

if __name__ == "__main__":
    main()
