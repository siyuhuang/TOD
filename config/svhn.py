
# Data
DATASET = 'svhn'
DATA_DIR = 'data/svhn'
CLASS = 10
NUM_TRAIN = 73257 
BATCH = 128   
SUBSET = 29302 
START = 7324 
ADDENDUM  = 3662 

# Active learning setting
TRIALS = 3
CYCLES = 7

# Training setting
MARGIN = 1.0  
WEIGHT = 0.005
EPOCH = 200
LR = 0.1    
MOMENTUM = 0.9
WDECAY = 5e-4
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model