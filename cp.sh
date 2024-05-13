git pull
scons Werror=1 debug=0 asserts=1 neon=1 opencl=1 examples=1 os=linux arch=armv8a -j4
cp build/libarm_compute* /home/run/lib 
cp build/examples/graph_bert_base_uncased /home/run 
cp build/examples/graph_bert_large_uncased /home/run