
# Data
DATASET = 'cifar100'
DATA_DIR = 'data'
CLASS = 100
NUM_TRAIN = 50000 
BATCH = 128   
SUBSET = 25000 
START = 5000 
ADDENDUM  = 2500 

# Active learning setting
TRIALS = 3
CYCLES = 7

# Training setting
MARGIN = 1.0  
WEIGHT = 0.05
EPOCH = 200
LR = 0.1    
MOMENTUM = 0.9
WDECAY = 5e-4
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model