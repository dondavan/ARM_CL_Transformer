export LD_LIBRARY_PATH=lib/
./graph_bert_base_uncased --threads=4 --text=./data/input_text.txt --segment=./data/input_segment.txt --data=./data/bert-base-uncased_npy/ --vocabulary=./data/vocab/bert-base-uncased_vocab.txt --raw-output=true
rm measure_output.txt