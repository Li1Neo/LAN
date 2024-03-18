export CUDA_VISIBLE_DEVICES=0
python run.py --epoch 10 --classification_level action --device cuda:0  --batch_size 512 --encoder_type lstm --lr 1e-5 --srcroot data/r5.2 --mixloss --graph_metric_type weighted_cosine --OverSampling 0.1 --add_graph_regularization --reduction lastpositionattention --use_sample_weight --hard_negative_weight 10
