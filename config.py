# RRDB
nf = 3
gc = 32

# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.0
lr = 10 ** log10_lr
lr3 = 10 ** -5.0
epochs = 500
weight_decay = 1e-5
init_scale = 0.01

device_ids = [0]
single_device = True

# Super loss
lamda_reconstruction_1 = 2
lamda_reconstruction_2 = 2
lamda_reconstruction_3 = 2
lamda_guide_1 = 1
lamda_guide_2 = 1
lamda_guide_3 = 1
lamda_z_1 = 1
lamda_z_2 = 1
lamda_z_3 = 1

# Super Key
lamda_overall_key = 1
lamda_part_key = 1

# Train:
batch_size = 12
cropsize = 128
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.1

# Val:
cropsize_val_paris = 128
cropsize_val_imagenet = 256

cropsize_test_paris = 128
cropsize_test_imagenet = 256

batchsize_val = 3
shuffle_val = False
val_freq = 1

batchsize_test = 3
shuffle_test = False

# Dataset
Dataset_mode = 'PARIS'  # PARIS / ImageNet
Dataset_VAL_mode = 'PARIS'  # PARIS / ImageNet
Dataset_TEST_mode = 'PARIS'

TRAIN_PATH_PARIS = ''
VAL_PATH_PARIS = ''
TEST_PATH_PARIS = ''

TRAIN_PATH_IMAGENET = ''
VAL_PATH_IMAGENET = ''
TEST_PATH_IMAGENET = ''

# Saving checkpoints:
MODEL_PATH1 = 'model1/'
MODEL_PATH2 = 'model2/'
MODEL_PATH3 = 'model3/'
checkpoint_on_error = True
SAVE_freq = 5

TEST_PATH1 = 'image1/'
TEST_PATH2 = 'image2/'
TEST_PATH3 = 'image3/'

TEST_PATH1_cover = TEST_PATH1 + 'cover/'
TEST_PATH1_secret_1 = TEST_PATH1 + 'secret_1/'
TEST_PATH1_steg_1 = TEST_PATH1 + 'steg_1/'
TEST_PATH1_secret_rev_1 = TEST_PATH1 + 'secret-rev_1/'

TEST_PATH2_cover = TEST_PATH2 + 'cover/'
TEST_PATH2_secret_1 = TEST_PATH2 + 'secret_1/'
TEST_PATH2_secret_2 = TEST_PATH2 + 'secret_2/'
TEST_PATH2_steg_1 = TEST_PATH2 + 'steg_1/'
TEST_PATH2_steg_2 = TEST_PATH2 + 'steg_2/'
TEST_PATH2_secret_rev_1 = TEST_PATH2 + 'secret-rev_1/'
TEST_PATH2_secret_rev_2 = TEST_PATH2 + 'secret-rev_2/'

TEST_PATH3_cover = TEST_PATH3 + 'cover/'
TEST_PATH3_secret_1 = TEST_PATH3 + 'secret_1/'
TEST_PATH3_secret_2 = TEST_PATH3 + 'secret_2/'
TEST_PATH3_secret_3 = TEST_PATH3 + 'secret_3/'
TEST_PATH3_steg_1 = TEST_PATH3 + 'steg_1/'
TEST_PATH3_steg_2 = TEST_PATH3 + 'steg_2/'
TEST_PATH3_steg_3 = TEST_PATH3 + 'steg_3/'
TEST_PATH3_secret_rev_1 = TEST_PATH3 + 'secret-rev_1/'
TEST_PATH3_secret_rev_2 = TEST_PATH3 + 'secret-rev_2/'
TEST_PATH3_secret_rev_3 = TEST_PATH3 + 'secret-rev_3/'

# Load:
suffix_load = 'model_checkpoint_00000'
train_next = False

trained_epoch = 0

pretrain = False
PRETRAIN_PATH1 = 'model1/'
PRETRAIN_PATH2 = 'model2/'
PRETRAIN_PATH3 = 'model3/'
suffix_pretrain = 'model_checkpoint_00000'

