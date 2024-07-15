git pull
scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 examples=1 os=linux arch=armv8a -j4
cp build/libarm_compute* /home/run/lib 
sshpass -p 'khadas' scp build/libarm_compute* Khadas.local:/home/khadas/run/lib 
sshpass -p 'khadas' scp build/examples/graph_bert_base_uncased_CL Khadas.local:/home/khadas/run
sshpass -p 'khadas' scp build/examples/graph_bert_large_uncased_CL Khadas.local:/home/khadas/run
sshpass -p 'khadas' scp build/examples/graph_bert_base_uncased Khadas.local:/home/khadas/run
sshpass -p 'khadas' scp build/examples/graph_bert_large_uncased Khadas.local:/home/khadas/run