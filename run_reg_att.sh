# train model on coco dataset
# first train lm: 1e-4 to convergence(not finetune cnn), then 5e-5,start finetune cnn with lr 1e-5
if [ 1 -eq 1 ]; then 
   CUDA_VISIBLE_DEVICES=1 th train_reg_on_att.lua \
    -input_h5  ../coco_data/cocotalk.h5 \
    -input_json  ../coco_data/cocotalk.json \
    -rnn_size 512 \
    -word_encoding_size 300 \
    -image_encoding_size 300 \
    -attention_size 300 \
    -glove_path ../glove_word2vec/glove.6B.300d.txt \
    -start_from ./model_id.t7 \
    -optim_state_from ./optim_id.t7 \
    -glove_dim 300 \
    -batch_size 48 \
    -optim adam \
    -learning_rate 5e-5 \
    -optim_alpha 0.8 \
    -optim_beta 0.999 \
    -finetune_cnn_after 0 \
    -cnn_learning_rate 1e-5 \
    -val_images_use 5000  
fi
#-start_from ./model_id.t7 \
#-optim_state_from ./optim_id.t7 \
#    -use_glove true \
#    -glove_path ../glove_word2vec/glove.6B.300d.txt \
#    -glove_dim 300 i\
#     -start_from ./checkpoints_48/model_id.t7 \
#   -optim_state_from ./checkpoints_48/optim_id_latest.t7 \
