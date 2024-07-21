export LD_LIBRARY_PATH=lib/
vector='128'
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_0.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_1.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_2.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_3.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_4.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_5.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_6.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_7.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_8.txt
./graph_bert_base_uncased_CL --target=cl --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_vector_depth/${vector}_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
mv measure_output.txt measure_output_9.txt
mkdir ./thesis_result/data/bert-base-uncased_vector_depth/
mkdir ./thesis_result/data/bert-base-uncased_vector_depth/${vector}_npy/
mkdir ./thesis_result/data/bert-base-uncased_vector_depth/${vector}_npy/CL/
mv measure_output_* thesis_result/data/bert-base-uncased_vector_depth/${vector}_npy/CL/