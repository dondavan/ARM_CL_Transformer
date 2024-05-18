/*
 * Copyright (c) 2018-2020 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph/Graph.h"
#ifndef ARM_COMPUTE_GRAPH_GRAPH_PIPELINE_H
#define ARM_COMPUTE_GRAPH_GRAPH_PIPELINE_H

#include "arm_compute/graph/Graph.h"


namespace arm_compute
{
namespace graph
{
/** Graph class
 *
 * Represents a multiple source - multiple sink directed graph
 */
class GraphPipeline:public Graph
{
public:

	/*GraphPipeline(GraphID id, std::string name)
	    : _id(id), _name(std::move(name)), _nodes(), _edges(), _tensors(), _tagged_nodes(), _mtx()
	{
	}*/
	GraphPipeline(GraphID id, std::string name, char _PE, char _Host_PE, int _start, int _end)
		    : Graph(id, name), start_layer(_start), end_layer(_end), PE(_PE), Host_PE(_Host_PE)
	{
	}

private:
    int		start_layer;
    int		end_layer;
    char	PE;
    char	Host_PE;
};
}
}

#endif /* ARM_COMPUTE_GRAPH_GRAPH_PIPELINE_H */
