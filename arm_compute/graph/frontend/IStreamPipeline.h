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
#ifndef ARM_COMPUTE_GRAPH_ISTREAM_PIPELINE_H
#define ARM_COMPUTE_GRAPH_ISTREAM_PIPELINE_H

#include "arm_compute/graph/frontend/IStream.h"
#include <unordered_set>

#include <regex>

namespace arm_compute
{
namespace graph
{
// Forward declarations
class Graph;

namespace frontend
{
// Forward declarations
class ILayer;



class NodeMap{
public:
    void insert(std::pair<NodeID,int> key, std::pair<NodeID,int> value ){
        if (mm.find(key)!=mm.end()){
            auto vv=mm[key];
            bool exist=false;
            for(auto v:vv){
                if (v==value){
                    std::cerr<<"Already exist!\n";
                    exist=true;
                }
            }
            if(!exist){
                mm[key].push_back(value);
            }
        }
        else{
            std::vector<std::pair<NodeID,int>> vv;
            vv.push_back(value);
            mm.insert(std::make_pair(key,vv) );
        }
    }
    //There are three cases:
    //1- The source node (key) is not in mapping yet, which means we need to connect a T node to that node in source graph(key.second) and create a R node in the target subgraph and add the mapping to the records of mapping
    //2- The source node (key) is in mapping which means already a T node is added to the source node and R node in a subgraph:
    // a) if the R node is in this target graph we just need to move tail node to this R node instead of source node
    // b) if the R node is not in this subgraph, we need to add a R node in this subgraph and move tail node to this R instead of source node and add the mapping to the records
    std::pair<int,int> find(std::pair<NodeID,int> key, int target_graph){
        std::pair<NodeID,int> r={0,-2};
        //There is no mapping for this node-graph, so create T and R in source and destination subgraphs and add them to the mapping records
        if(mm.find(key)==mm.end()){
            //std::cerr<<"There is no mapping for node graph \n";
            r={0,-1};
            //create a T node and append to the node key.first in graph key.second (add it to mapping also)
            //create a R node in new graph (and add it to mapping)
        }
        //There is a record for this NodeID-GraphID so it already created the T and R nodes and are added to mapping records
        else{
            auto maps=mm[key];
            bool exist=false;
            //If there is a mapped node from that source node that is in this target subgraph
            for(auto v:maps){
                if(v.second==target_graph){
                    //std::cerr<<"There is a mapped node in this graph\n";
                    exist=true;
                    r=v;
                    //change the tail node from key.first to v.first
                    break;
                }
            }
            //If the there is no mapped node from the source node in this subgraph (but there is already a mapping which means the R node existed on that subgraph)
            //Then we need to create a R node in this subgraph and connect it to that T node
            if(!exist){
                for(auto v:maps){
                    if(v.second==key.second){
                        //std::cerr<<"The T node in that graph is node: "<<v.first<<"\n";
                        r=v;
                        //create a R node in new graph (and add it to mapping)
                        //add R node into the T node(v.first) of origin graph
                        break;
                    }
                }
            }
        }
        //std::cerr<<"mappd node for node "<<key.first<<" in graph "<<key.second<<" is node "<<r.first<<" in graph "<<r.second<<std::endl;
        return r;
    }
    std::pair<int,int> find(std::pair<NodeID*,int*> key, int target_graph){
    	return find(std::make_pair(*(key.first), *(key.second)), target_graph);
    }


    void print(){
        for(auto entry:mm){
            std::cerr<<"\n\n\n"<<entry.first.first<<" in graph "<<entry.first.second<<std::endl;
            for(auto v: entry.second){
                std::cerr<<"equals to: "<<v.first<<" in graph "<<v.second<<std::endl;
            }
        }
    }


private:
    //A mapping that records the mapping of NodeID-GraphID to antoher NodeID-GraphID (mapping of a node in a subgraph to another node in another subgraph)
    //So that when input node of a layer is a node in another subgraph we should create a node in this subgraph(R node) and add a node to that node(T node) to send data from that subgraph to
    //the R in this subgraph. So, instead of that node we use R node in this subgraph which have same data
    std::map< std::pair<NodeID,int> , std::vector< std::pair<NodeID,int>> > mm;
};




/** Stream interface **/
class IStreamPipeline: public IStream
{
public:
    virtual ~IStreamPipeline() = default;
    /** Adds a layer to the stream
     *
     * @param[in] layer Layer to add
     */
    virtual void add_layer(ILayer &layer) = 0;
    /** Returns the underlying graph
     *
     * @return Underlying graph
     */
    virtual Graph &graph() = 0;
    /** Returns the underlying graph
     *
     * @return Underlying graph
     */
    virtual const Graph &graph() const = 0;
    /** Returns the tail node of the Stream
     *
     * @return Tail Node ID
     */
    NodeID tail_node()
    {
    	//std::cerr<<"ISTREAMPipeline callin tail_node() "<<_tail_node<<std::endl;
        //return _tail_node;
    	auto _n=node_map.find(std::make_pair(_tail_node, tail_graph_id), _target_graph);
		if (_n.second==_target_graph){
			tail_graph_id=_target_graph;
			_tail_node=_n.first;
			return _n.first;
		}
		else{
			return _tail_node;
		}
    }

    NodeID tail_node(int target_graph){
    	auto _n=node_map.find(std::make_pair(_tail_node, tail_graph_id), target_graph);
    	if (_n.second==target_graph){
    		tail_graph_id=target_graph;
    		_tail_node=_n.first;
    		return _n.first;
    	}
    	else{
    		//std::cerr<<"\n\n\n\n\nERROR! There is no node in target graph for tail node\n\n\n\n";
    		return _tail_node;
    	}

    }

    /** Returns the stream hints that are currently used
     *
     * @return Stream hints
     */
    virtual StreamHints &hints()
    {
    	std::string s;
		//s="__calling hints in IstreamPipeline is "+ std::to_string((int)(_hints.target_hint)) +"\n";
		//std::cerr<<s;
        return _hints;
    }
    /** Forwards tail of stream to a given nid
     *
     * @param[in] nid NodeID of the updated tail node
     */
    virtual void forward_tail(NodeID nid)
    {
        _tail_node = (nid != NullTensorID) ? nid : _tail_node;
    }
    //virtual StreamHints &hints();

    //IStreamPipeline & operator<<(ILayer &layer);
    //IStreamPipeline & operator<<(ILayer &&layer);
    //virtual void next_layer();
    /*NodeID* get_tail_p(){
    	return &_tail_node;
    }
    int* get_graph_id_p(){
    	return &graph_id;
    }*/
    int get_tail_graph_id() override{
    	return tail_graph_id;
    }
    std::pair<NodeID,int> get_position(){
    	return std::make_pair(this->tail_node(),this->get_tail_graph_id());
    }

    int target_graph(int layer){
    	if(start_layer.size()==0){
    		return 0;
    	}
    	for(int i=0; i<start_layer.size(); i++){
    		if(layer>=start_layer[i] && layer<=end_layer[i]){
    			return i;
    		}
    	}
    	return start_layer.size()-1;
    	//return -1;
    }

    static bool is_next_layer(std::string name);
    static bool is_end_layer(std::string name);

    inline static std::string						graph_name;
    //inline static std::unordered_set<std::string> ending_tasks;

protected:
    inline static int 			current_layer={0};

    inline static int			_target_graph={0};
    int		tail_graph_id=0;
    inline static NodeMap				node_map;
    inline static std::vector<int>	start_layer;
    inline static std::vector<int>	end_layer;



    //StreamHints _hints     = {};              /**< Execution and algorithmic hints */
    //NodeID      _tail_node = { EmptyNodeID }; /**< NodeID pointing to the last(tail) node of the graph */
};
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ISTREAM_H */
