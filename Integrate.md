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

runtime finished -> src

= src/graph/algorithms/TopologicalSort.cpp
= src/graph/backends/CL/CLTensorHandle.cpp
= src/graph/backends/NEON/NEDeviceBackend.cpp
= src/graph/backends/NEON/NETensorHandle.cpp

+ src/graph/backends/NPU/*

= src/graph/detail/CrossLayerMemoryManagerHelpers.cpp
= src/graph/detail/ExecutionHelpers.cpp