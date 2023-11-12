==>>the script to train Bart-InterMRC(base), where the evidence in the dateset is annotated by the STS(T5-LR-DA(B)) model
python BartInterMRC.py \
--is_new \
--is_training \
--model_level facebook/bart-base \
--mode t5-base \
--learning_rate 1e-4 \
--mini_batch_size 10 \
--left_threshold 0.2 \
--right_threshold 0.3 \
--num 2 \
--device cuda:0 

==>>the script to train Bart-InterMRC(base) with pseudo datasets
python BartInterMRC.py \
--is_training \
--model_level facebook/bart-base \
--mode pseudo \
--learning_rate 1e-4 \
--mini_batch_size 10 \
--left_threshold 0.01 \
--right_threshold 0.2 \
--num 2\
--device cuda:0 

==>>the script to train Bart-InterMRC(large) with pseudo datasets
python BartInterMRC.py \
--is_training \
--model_level facebook/bart-large \
--mode pseudo \
--learning_rate 1e-4 \
--mini_batch_size 5 \
--device cuda:0
