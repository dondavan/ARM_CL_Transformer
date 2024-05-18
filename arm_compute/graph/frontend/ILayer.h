/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_ILAYER_H
#define ARM_COMPUTE_GRAPH_ILAYER_H

//Ehsan
#include "arm_compute/graph/frontend/SubStream.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
// Forward declarations
class IStream;

/** ILayer interface */
class ILayer
{
public:
    /** Default destructor */
    virtual ~ILayer() = default;
    /** Create layer and add to the given stream.
     *
     * @param[in] s Stream to add layer to.
     *
     * @return ID of the created node.
     */
    virtual NodeID create_layer(IStream &s) = 0;
    /** Sets the name of the layer
     *
     * @param[in] name Name of the layer
     *
     * @return The layer object
     */
    ILayer &set_name(std::string name)
    {
        _name = name;
        return *this;
    }
    /** Layer name accessor
     *
     * @return Returns the name of the layer
     */
    const std::string &name() const
    {
        return _name;
    }
    //Ehsan
    std::vector<std::pair<NodeID,int>>& get_input_nodes(){return _input_nodes;};
    void add_input_node(NodeID node, int graph_id){ _input_nodes.push_back(std::make_pair(node,graph_id));};
    void add_input_node_ptr(NodeID node, int graph_id){ _input_nodes.push_back(std::make_pair(node,graph_id));};

protected:
    std::string _name = {};
    //Ehsan
    std::vector<std::pair<NodeID,int>> _input_nodes;
};
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ILAYER_H */
