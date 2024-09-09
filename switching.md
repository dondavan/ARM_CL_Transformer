# New Device Backend - For Both NEON and CL
Approach:
    Add a new targert: switching
    Launch both NEON and CL device backend
Files:
    src/graph/GraphManager.cpp
    src/graph/backends/BackendRegistry.cpp
    src/graph/Utils.cpp: force_target_to_graph
                         setup_requested_backend_context
    src/graph/INode.cpp: set_assigned_target

# Add device execution target to layer:
Approach:
    Layer target -> workload
    When configured, use layer target.
    Otherwise use default target.
    Layer target -> workload
Files:
    src/graph/detail/ExecutionHelpers.cpp: configure_all_nodes

# Memory handler - Between NEON and CL
Problem:
    Differetn target backend has different tensor handle,
    Need to address interchange between NEON and CL tensor.
    Mapping after individual execution? or whole execution
Files:
    src/graph/Tensor.cpp: call_accessor(): 
                            _handle->map(true);
    ->
        src/graph/backends/NEON/NETensorHandle.cpp
        src/graph/backends/CL/CLTensorHandle.cpp

# Starting from worktask:

    ExecutionWorkload contains: ExecutionTask
    add target tag to task(from node description):
        src/graph/detail/ExecutionHelpers.cpp: configure_all_nodes
                node->assigned_target();
                
    schedule different operation
    handle different memory

 