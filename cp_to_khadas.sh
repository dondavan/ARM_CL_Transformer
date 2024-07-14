sshpass -p 'khadas' scp build/libarm_compute* Khadas.local:/home/khadas/run/lib 
sshpass -p 'khadas' scp build/examples/graph_bert_base_uncased_CL Khadas.local:/home/khadas/run
sshpass -p 'khadas' scp build/examples/graph_bert_large_uncased_CL Khadas.local:/home/khadas/run
sshpass -p 'khadas' scp build/examples/graph_bert_base_uncased Khadas.local:/home/khadas/run
sshpass -p 'khadas' scp build/examples/graph_bert_large_uncased Khadas.local:/home/khadas/run