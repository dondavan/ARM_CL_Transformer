= means modified
+ means add

+ Libs/*

+ NPU/*

= arm_compute/graph/backends/CL/CLTensorHandle.h
= arm_compute/graph/backends/NEON/NETensorHandle.h
+ arm_compute/graph/backends/NPU/*
+ arm_compute/graph/backends/FunctionHelpers.h:create_npu_function

+ arm_compute/graph/detail/ExecutionHelpersPipeline.h

= arm_compute/graph/frontend/ILayer.h
= arm_compute/graph/frontend/IStream.h
+ arm_compute/graph/frontend/IStreamPipeline.h
= arm_compute/graph/frontend/Layers.h
= arm_compute/graph/frontend/Stream.h
+ arm_compute/graph/frontend/StreamPipeline.h
= arm_compute/graph/frontend/SubStream.h

+ arm_compute/graph/nodes/EarlyExitOutputNode.h
+ arm_compute/graph/nodes/NPUNode.h
= arm_compute/graph/nodes/Nodes.h
+ arm_compute/graph/nodes/ReceiverNode.h
+ arm_compute/graph/nodes/SenderNode.h

= arm_compute/graph/Graph.h
= arm_compute/graph/GraphBuilder.h
= arm_compute/graph/GraphManager.h
+ arm_compute/graph/GraphManagerPipeline.h
+ arm_compute/graph/GraphPipeline.h
= arm_compute/graph/ITensorHandle.h
= arm_compute/graph/Tensor.h
+ arm_compute/graph/TensorPipeline.h
= arm_compute/graph/TypePrinter.h
= arm_compute/graph/Types.h
= arm_compute/graph/Workload.h

graph finished -> runtime

= arm_compute/runtime/CPP/CPPScheduler.h
+ arm_compute/runtime/NPU/*
= arm_compute/runtime/IScheduler.h
= arm_compute/runtime/Scheduler.h

runtime finished -> src

= src/graph/algorithms/TopologicalSort.cpp
= src/graph/backends/CL/CLTensorHandle.cpp
= src/graph/backends/NEON/NEDeviceBackend.cpp
= src/graph/backends/NEON/NETensorHandle.cpp

+ src/graph/backends/NPU/*

= src/graph/detail/CrossLayerMemoryManagerHelpers.cpp
= src/graph/detail/ExecutionHelpers.cpp
+ src/graph/detail/ExecutionHelpersPipeline.cpp

+ src/graph/frontend/IStreamPipeline.cpp
+ src/graph/frontend/StreamPipeline.cpp
= src/graph/frontend/Stream.cpp
= src/graph/frontend/SubStream.cpp

+ src/graph/nodes/EarlyNodeOutputNode.cpp
+ src/graph/nodes/NPUNode.cpp
+ src/graph/nodes/ReceiverNode.cpp
+ src/graph/nodes/SenderNode.cpp

= src/graph/GraphBuilder.cpp
= src/graph/GraphManager.cpp
+ src/graph/GraphManagerPipeline.cpp
= src/graph/Tensor.cpp
+ src/graph/TensorPipeline.cpp
= src/graph/Utils.cpp
= src/graph/Workload.cpp

= src/runtime/CPP/CPPScheduler.cpp

+ src/runtime/NPU/*
src/runtime/Scheduler.cpp
src/runtime/SchedulerUtils.cpp

src finished -> utils
+ utils/B/*
+ utils/DVFS/*
+ utils/n-pipe/*
+ utils/n-pipe-NPU/*
+ utils/pipe-all/*
+ utils/Power/*

= utils/CommonGraphOptions.cpp
= utils/CommonGraphOptions.h

= utils/GraphUtils.cpp
= utils/GraphUtils.h

+ utils/GraphUtilsPipeline.cpp
+ utils/GraphUtilsPipeline.h
= utils/Utils.h

+ utils/UtilsPipeline.h
+ utils/UtilsPipeline.cpp
+ utils/main_layer_checker.h
+ utils/ttt.cpp



****************************************************************************
= arm_compute/graph/nodes/SenderNode.h
= arm_compute/graph/nodes/ReceiverNode.h
= arm_compute/graph/Tensor.h
= arm_compute/graph/TensorPipeline.h
= arm_compute/graph/frontend/IStreamPipeline.h
= arm_compute/graph/frontend/IStream.h          Hints
= src/graph/GraphManager.cpp