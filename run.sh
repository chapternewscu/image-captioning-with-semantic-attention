# train model on coco dataset  
if [ 1 -eq 1 ]; then 
   CUDA_VISIBLE_DEVICES=0 th train.lua \
    -input_h5  ../coco_data/cocotalk.h5 \
    -input_json  ../coco_data/cocotalk.json \
    -rnn_size 512 \
    -word_encoding_size 300 \
    -image_encoding_size 300 \
    -attention_size 300 \
    -glove_path ../glove_word2vec/glove.6B.300d.txt \
    -glove_dim 300 \
    -batch_size 16 \
    -optim adam \
    -learning_rate 1e-4 \
    -optim_alpha 0.8 \
    -optim_beta 0.999 \
    -cnn_learning_rate 1e-5 \
    -val_images_use 5000  
fi
#-start_from ./model_id.t7 \
#-optim_state_from ./optim_id.t7 \
#    -use_glove true \
#    -glove_path ../glove_word2vec/glove.6B.300d.txt \
#    -glove_dim 300 \
