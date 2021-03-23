# 09/15
python train_net.py --cfg_file configs/city_ct_rcnn.yaml model rcnn_det

python train_net.py --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake det_model rcnn_det
### 09-23: 3 snake loss
python train_net.py --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake_loss3 det_model rcnn_det

python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake

# 09/16
python train_net.py --cfg_file configs/city_ct_rcnn.yaml model rcnn_det2
python train_net.py --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake2 det_model rcnn_det2
python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake2 gpus 1,

# 09/17
python train_net.py --cfg_file configs/city_ct_rcnn.yaml model rcnn_det3
python train_net.py --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake3 det_model rcnn_det3
python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake3 gpus 2,

# 09/20
### because part of previous data is discarded
python train_net.py --cfg_file configs/city_ct_rcnn.yaml model rcnn_det_ori_data
python train_net.py --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake_ori_data det_model rcnn_det_ori_data
python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake_ori_data gpus 2,

# 09/22
python train_net.py --cfg_file configs/city_ct_rcnn.yaml model rcnn_det_ori_data2
python train_net.py --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake_ori_data2 det_model rcnn_det_ori_data2
python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake_ori_data2 gpus 1,


# 09/24
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model try100xloss det_model rcnn_det

# 09/26
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model together det_model no
python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model together det_model no

# 09/26 
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1x_avg_loss det_model rcnn_det2
-> re-train with longer 加长训练
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1x_avg_loss_300ep det_model rcnn_det2 train.epoch 300 train.milestones 120,180,220,250

# --- test
python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1x_avg_loss_300ep_test det_model rcnn_det2 segm_or_bbox bbox  --> 41.8

python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1x_avg_loss_300ep_test det_model rcnn_det2 segm_or_bbox segm  --> 36.1

python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1x_avg_loss_300ep_test det_model rcnn_det2 test.dataset CityscapesTest 

# --- speed
python run.py --type network --cfg_file configs/city_rcnn_snake.yaml 
<!-- -> 0.16250579967731382 s/img
0.16309948374585406
0.16216984803114481 -->

python run.py --type network --cfg_file configs/city_rcnn_dance.yaml model final_test
<!-- -> 0.1598713805035847 s/img
0.16132551043983398
0.16125324150411094 -->

# final submit
python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model final_test test.dataset CityscapesTest

python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model final_test_cp test.dataset CityscapesVal gpus 1,



# reproduced submit
python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake2 test.dataset CityscapesVal

python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml model rcnn_snake2 test.dataset CityscapesTest gpus 1,




# 09/28 
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1x_avg_loss det_model rcnn_det2

# 1.5x
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1p5x_avg_loss_300ep det_model rcnn_det2 train.epoch 300 train.milestones 120,180,220,250

# cosine
python train_net.py --cfg_file configs/city_rcnn_dance_cosine.yaml model no_scaling_1x_avg_loss_cosine det_model rcnn_det2

# 1.5 origin
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model no_scaling_1p5x_avg_loss_250ep det_model rcnn_det train.epoch 250

python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model final_test det_model rcnn_det2 test.dataset CityscapesVal



# 09/27 maintain scale，change coef and lr
python train_net.py --cfg_file configs/city_rcnn_dance.yaml model has_scaling_250x_avg_loss det_model rcnn_det2 train.lr 1e-3

# choose a good det model:
python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model no det_model rcnn_det segm_or_bbox bbox  --> 40.0


python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model no det_model rcnn_det2 segm_or_bbox bbox  -->  40.3

python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model no det_model rcnn_det3 segm_or_bbox bbox  --> 39.7

python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml model no det_model rcnn_det_ori_data segm_or_bbox bbox gpus 1, --> 39.0


### det after snake:
python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml task rcnn_snake model rcnn_snake det_model no segm_or_bbox bbox  --> 40.8






# ---------------------------------------------------- #
# -> pretrained: 52.+
python run.py --type evaluate --cfg_file configs/sbd_snake.yaml test.dataset SbdVal


# SBD
python train_net.py --cfg_file configs/sbd_snake.yaml model sbd_snake
python run.py --type evaluate --cfg_file configs/sbd_snake.yaml model sbd_snake_test det_model no test.dataset SbdVal --> 55.3

# ---
python train_net.py --cfg_file configs/sbd_dance.yaml model sbd_dance

python run.py --type evaluate --cfg_file configs/sbd_dance.yaml model sbd_dance_test det_model no test.dataset SbdVal --> 56.2
# ---

# speed
### snake
python run.py --type network --cfg_file configs/sbd_snake.yaml test.dataset SbdValSpeed  --> 0.02888488112341168

### dance
python run.py --type network --cfg_file configs/sbd_dance.yaml model sbd_dance_test test.dataset SbdValSpeed  --> 0.026112741261034352




# Visualization
python run.py --type visualize --cfg_file configs/sbd_dance.yaml model sbd_dance_test test.dataset SbdVal ct_score 0.3

python run.py --type visualize --cfg_file configs/sbd_snake.yaml  test.dataset SbdVal ct_score 0.3