import sys
sys.path.append('.')

import os
import torch

import pickle

import shutil

import torch
from torch.utils.data import TensorDataset

from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader, random_split

from tllib.vision.transforms import ResizeImage

from utils.utils import dump, load




def get_loaders(domain, config):
    """
    Returns the following 6 loaders
    - Full (train & test)
    - Retain (train & test)
    - Forget (only test)
    - Retain Subset (only train)

    """

    prune_domainnet(config)
    
    full_dataset = get_dataset(domain, config)
    retain_dataset = filter_dataset(full_dataset, domain, config['forget_classes'], config, exclude=True)

    full_train_dataset, full_test_dataset = random_split(full_dataset, lengths=[config['split'], 1-config['split']])
    retain_train_dataset, retain_test_dataset = random_split(retain_dataset, lengths=[config['split'], 1-config['split']])
    forget_dataset = filter_dataset(full_dataset, domain, config['forget_classes'], config, exclude=False)
    retain_subset = get_subset(retain_dataset, config)

    full_train_dl = DataLoader(full_train_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    full_test_dl = DataLoader(full_test_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=True)
    retain_train_dl = DataLoader(retain_train_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    retain_test_dl = DataLoader(retain_test_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=True)
    forget_dl = DataLoader(forget_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=False)
    retain_subset_dl = DataLoader(retain_subset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)

    return full_train_dl, full_test_dl, retain_train_dl, retain_test_dl, forget_dl, retain_subset_dl


def get_dataset(domain, config):
    """
    Returns the dataset

    """
    
    
    torch.manual_seed(config['seed'])

    RESIZE = 256
    SIZE = 224
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        ResizeImage(RESIZE),
        CenterCrop(SIZE),
        ToTensor(),
        Normalize(mean=MEANS, std=STDS)
    ])

    if config['dataset'] == 'OfficeHome':
        assert domain in ['Art', 'Clipart', 'Product', 'Real_World']
        config['num_classes'] = 65
        config['size'] = 224
        config['channels'] = 3

    elif config['dataset'] == 'DomainNet':
        assert domain in ['clipart', 'painting', 'real', 'sketch']    
        config['num_classes'] = 126
        config['size'] = 224
        config['channels'] = 3

    elif config['dataset'] == 'Office31':
        assert domain in ['amazon', 'dslr', 'webcam']
        config['num_classes'] = 31
        config['size'] = 224
        config['channels'] = 3
    
    path = os.path.join(config['data_path'], config['dataset'], domain)        
    dataset = ImageFolder(root=path, transform=transform)
        
    return dataset


def filter_dataset(dataset, domain, classes, config, exclude=False):
    """
    Filters Dataset based on classes

    """
    
    
    path = os.path.join(config['dump_path'], config['dataset'], domain, 'labels.p')

    if not os.path.exists(path):
        labels = []
        for _, label in tqdm(dataset, desc=f"Filtering {config['dataset']} {domain}"):
            labels.append(label)
        labels = torch.tensor(labels)
        dump(labels, path)
    
    labels = load(path)

    if exclude:
        mask = ~torch.isin(labels, torch.tensor(classes))
    else:
        mask = torch.isin(labels, torch.tensor(classes))

    indices = torch.nonzero(mask).squeeze().tolist()
    filtered_dataset = Subset(dataset, indices)

    return filtered_dataset


def get_subset(dataset, config):
    """
    Returns the subset of data accessable after training.
    
    Arbitrarily considering approximately 20 samples per class

    """

    subset_size = config['num_classes'] * 20
    subset_size = min(subset_size, len(dataset))
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, subset_indices)

    return subset

def prune_domainnet(config):
    """
    Convert to DomainNet-126    
    """ 

    if config['dataset'] != 'DomainNet':
        return

    unreq_domains = {'infograph', 'quickdraw'}

    req_domains = {'clipart', 'painting', 'sketch', 'real'}

    unreq_classes = {'rollerskates', 'frying_pan', 'van', 'tractor', 'ocean', 'belt', 'cooler', 
                     'swing_set', 'roller_coaster', 'brain', 'triangle', 'waterslide', 'jail', 
                     'The_Mona_Lisa', 'octopus', 'tree', 'lollipop', 'sweater', 'beach', 'apple', 
                     'steak', 'flying_saucer', 't-shirt', 'eraser', 'paintbrush', 'door', 'octagon', 
                     'hamburger', 'cloud', 'tornado', 'stereo', 'syringe', 'wristwatch', 'smiley_face', 
                     'wine_glass', 'hospital', 'animal_migration', 'paper_clip', 'bicycle', 'garden', 
                     'hockey_puck', 'washing_machine', 'parrot', 'scorpion', 'hourglass', 'diving_board', 
                     'fan', 'house_plant', 'megaphone', 'zigzag', 'sock', 'book', 'camouflage', 'floor_lamp', 
                     'leg', 'hot_air_balloon', 'church', 'envelope', 'rain', 'birthday_cake', 
                     'remote_control', 'owl', 'sun', 'motorbike', 'hedgehog', 'snorkel', 'spreadsheet',
                       'paint_can', 'ladder', 'skull', 'flip_flops', 'hockey_stick', 'knee', 'beard', 
                       'tent', 'baseball_bat', 'ambulance', 'skyscraper', 'bush', 'snowman', 'passport', 
                       'saw', 'lighter', 'telephone', 'stitches', 'snowflake', 'snail', 'hot_dog', 
                       'drill', 'sink', 'police_car', 'shovel', 'basketball', 'broom', 'trumpet', 
                       'windmill', 'microwave', 'stop_sign', 'pond', 'dresser', 'cookie', 'angel', 
                       'boomerang', 'mailbox', 'toaster', 'toothpaste', 'parachute', 'calendar', 'bread', 
                       'car', 'donut', 'face', 'bandage', 'traffic_light', 'diamond', 'map', 'necklace', 
                       'couch', 'flashlight', 'barn', 'firetruck', 'lighthouse', 'pickup_truck', 'hexagon', 
                       'bowtie', 'pliers', 'spoon', 'hurricane', 'wine_bottle', 'marker', 'square', 'moustache', 
                       'violin', 'sword', 'toothbrush', 'tooth', 'harp', 'pizza', 'popsicle', 'golf_club', 
                       'picture_frame', 'arm', 'moon', 'mouth', 'light_bulb', 'suitcase', 'bench', 'stove', 
                       'sailboat', 'eye', 'piano', 'airplane', 'bracelet', 'pool', 'yoga', 'oven', 'circle', 
                       'key', 'clock', 'radio', 'knife', 'sleeping_bag', 'ice_cream', 'crayon', 'cup', 'bridge', 
                       'fire_hydrant', 'stethoscope', 'tennis_racquet', 'pants', 'wheel', 'clarinet', 'underwear', 
                       'binoculars', 'mermaid', 'bed', 'garden_hose', 'bat', 'squiggle', 'toilet', 'hot_tub', 'hat', 
                       'nose', 'line', 'postcard', 'school_bus', 'shorts', 'keyboard', 'bucket', 'soccer_ball', 'lightning', 
                       'hand', 'dishwasher', 'crown', 'fireplace', 'elbow', 'sandwich', 'star', 'baseball', 'stairs', 'mountain', 
                       'rake', 'shark', 'campfire', 'backpack', 'rainbow', 'house', 'ear', 'river', 'nail', 'bulldozer', 'grass', 
                       'trombone', 'palm_tree', 'finger', 'jacket', 'headphones', 'matches', 'scissors'}
    
    root_path = os.path.join(config['data_path'], 'DomainNet')
    
    for domain in unreq_domains:
        path = os.path.join(root_path, domain)
        if os.path.exists(path):
            shutil.rmtree(path)

    for domain in req_domains:
        for class_ in unreq_classes:
            path = os.path.join(root_path, domain, class_)
            if os.path.exists(path):
                shutil.rmtree(path)

