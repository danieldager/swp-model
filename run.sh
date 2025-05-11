#!/bin/bash
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s1_sn_ec --checkpoint 75 --ablate e40; 
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s4_sn_ec --checkpoint 96 --ablate e112 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s5_sn_ec --checkpoint 93 --ablate e42 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s37_sn_ec --checkpoint 94 --ablate e68 ;
# python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s42_sn_ec --checkpoint 75 --ablation e16 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s45_sn_ec --checkpoint 58 --ablate e34 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s68_sn_ec --checkpoint 95 --ablate e44 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s70_sn_ec --checkpoint 92 --ablate e95 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s83_sn_ec --checkpoint 56 --ablate e58 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s95_sn_ec --checkpoint 93 --ablate e42 ;
python scripts/test_repetition.py --model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 --train_name b1024_l0.001_fall_s96_sn_ec --checkpoint 82 --ablate e30 ;

