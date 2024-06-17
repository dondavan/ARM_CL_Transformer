git pull
scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 examples=1 os=linux arch=armv8a -j4
scp build/libarm_compute* Khadas.local:/home/khadas/run/lib 
scp build/examples/graph_bert_base_uncased Khadas.local:/home/khadas/run
scp build/examples/graph_bert_large_uncased Khadas.local:/home/khadas/run