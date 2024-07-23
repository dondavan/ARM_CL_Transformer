export LD_LIBRARY_PATH=lib/

mkdir ./thesis_result/data/bert-base-uncased_heads/

mkdir ./thesis_result/data/bert-base-uncased_heads/h1/
mkdir ./thesis_result/data/bert-base-uncased_heads/h1/NEON/
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h1 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h1/NEON/

mkdir ./thesis_result/data/bert-base-uncased_heads/h2/
mkdir ./thesis_result/data/bert-base-uncased_heads/h2/NEON/
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h2 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h2/NEON/

mkdir ./thesis_result/data/bert-base-uncased_heads/h4/
mkdir ./thesis_result/data/bert-base-uncased_heads/h4/NEON/
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h4 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h4/NEON/

mkdir ./thesis_result/data/bert-base-uncased_heads/h6/
mkdir ./thesis_result/data/bert-base-uncased_heads/h6/NEON/
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h6 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h6/NEON/

mkdir ./thesis_result/data/bert-base-uncased_heads/h8/
mkdir ./thesis_result/data/bert-base-uncased_heads/h8/NEON/
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h8 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h8/NEON/

mkdir ./thesis_result/data/bert-base-uncased_heads/h12/
mkdir ./thesis_result/data/bert-base-uncased_heads/h12/NEON/
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h12 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h12/NEON/

mkdir ./thesis_result/data/bert-base-uncased_heads/h16/
mkdir ./thesis_result/data/bert-base-uncased_heads/h16/NEON/
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h16 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h16/NEON/

mkdir ./thesis_result/data/bert-base-uncased_heads/h24/
mkdir ./thesis_result/data/bert-base-uncased_heads/h24/NEON/
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_h24 --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mv measure_output_* ./thesis_result/data/bert-base-uncased_heads/h16/NEON/