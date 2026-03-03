#pragma once 

#include <vector>
#include <stdexcept>
#include <iterator>
#include <cstddef>

#include <torch/data.h>
#include <torch/data/dataloader.h>

struct GraphSample 
{
    torch::Tensor edge_index;     // [2, num_edges]
    torch::Tensor node_attr;      // [num_nodes, input_size]
    torch::Tensor node_hit_id;    // [num_nodes, 1]
    torch::Tensor edge_attr;      // [num_edges, edge_attr_size]
    torch::Tensor answer;         // [num_edges, 1]
    torch::Tensor new_edge_start; // [1, 1]

    GraphSample() = default;

    GraphSample(torch::Tensor edge_index_,
                torch::Tensor node_attr_,
                torch::Tensor node_hit_id_,
                torch::Tensor edge_attr_) :
                edge_index(std::move(edge_index_)),
                node_attr(std::move(node_attr_)),
                node_hit_id(std::move(node_hit_id_)),
                edge_attr(std::move(edge_attr_)),
                answer(torch::empty({0, 1})),
                new_edge_start(torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kInt32))) {}
    
    GraphSample(torch::Tensor edge_index_,
                torch::Tensor node_attr_,
                torch::Tensor node_hit_id_,
                torch::Tensor edge_attr_,
                torch::Tensor answer_) :
                edge_index(std::move(edge_index_)),
                node_attr(std::move(node_attr_)),
                node_hit_id(std::move(node_hit_id_)),
                edge_attr(std::move(edge_attr_)),
                answer(std::move(answer_)),
                new_edge_start(torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kInt32))) {}

    GraphSample(torch::Tensor edge_index_,
                torch::Tensor node_attr_,
                torch::Tensor node_hit_id_,
                torch::Tensor edge_attr_,
                torch::Tensor answer_,
                torch::Tensor new_edge_start_) :
                edge_index(std::move(edge_index_)),
                node_attr(std::move(node_attr_)),
                node_hit_id(std::move(node_hit_id_)),
                edge_attr(std::move(edge_attr_)),
                answer(std::move(answer_)),
                new_edge_start(new_edge_start_) {}

    GraphSample(const GraphSample& other) = default;
    GraphSample(GraphSample&& other) noexcept = default;

    virtual ~GraphSample() = default;
    
    GraphSample& operator=(const GraphSample& other) = default;
    GraphSample& operator=(GraphSample&& other) noexcept = default;

    virtual GraphSample& to(torch::ScalarType Dtype, torch::ScalarType Itype) 
    {
        edge_index     = edge_index.to(Itype);
        node_attr      = node_attr.to(Dtype);
        node_hit_id    = node_hit_id.to(Itype);
        edge_attr      = edge_attr.to(Dtype);
        answer         = answer.to(Dtype);
        new_edge_start = new_edge_start.to(Itype);
        
        return *this;
    }

    virtual GraphSample& to(torch::Device device) 
    {
        edge_index     = edge_index.to(device);
        node_attr      = node_attr.to(device);
        node_hit_id    = node_hit_id.to(device);
        edge_attr      = edge_attr.to(device);
        answer         = answer.to(device);
        new_edge_start = new_edge_start.to(device);
        
        return *this;
    }

    static GraphSample batch(const std::vector<GraphSample>& samples)
    {
        std::vector<torch::Tensor> edge_indices, node_attrs, node_hit_ids, edge_attrs, answers;
        torch::Tensor new_edge_start = torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kInt32));

        int64_t node_offset = 0;
        int64_t hit_offset = 0;
        
        for (const auto& s : samples)
        {
            if (s.new_edge_start.item<int>() == 0)
            {
                edge_indices.push_back(s.edge_index + node_offset);
                edge_attrs.push_back(s.edge_attr);
                answers.push_back(s.answer);
            }
            else
            {
                edge_indices.push_back(s.edge_index.index({torch::indexing::Ellipsis,
                    torch::indexing::Slice(s.new_edge_start.item<int>(), torch::indexing::None)}) + node_offset);
                edge_indices.insert(edge_indices.begin(), s.edge_index.index({torch::indexing::Ellipsis,
                    torch::indexing::Slice(0, s.new_edge_start.item<int>())}) + node_offset);
                edge_attrs.push_back(s.edge_attr.index({torch::indexing::Slice(s.new_edge_start.item<int>(), torch::indexing::None)}));
                edge_attrs.insert(edge_attrs.begin(), s.edge_attr.index({torch::indexing::Slice(0, s.new_edge_start.item<int>())}));
                answers.push_back(s.answer.index({torch::indexing::Slice(s.new_edge_start.item<int>(), torch::indexing::None)}));
                answers.insert(answers.begin(), s.answer.index({torch::indexing::Slice(0, s.new_edge_start.item<int>())}));
            }
            node_attrs.push_back(s.node_attr);
            node_hit_ids.push_back(s.node_hit_id + hit_offset);
            new_edge_start += s.new_edge_start;

            node_offset += s.node_attr.size(0);
            hit_offset += s.node_hit_id.max().item<int64_t>();
        }

        return GraphSample{torch::cat(edge_indices, 1),
                           torch::cat(node_attrs, 0),
                           torch::cat(node_hit_ids, 0),
                           torch::cat(edge_attrs, 0),
                           torch::cat(answers, 0),
                           new_edge_start};
    }
};

template <typename SampleType = GraphSample>
class GraphDataset : public torch::data::datasets::Dataset<GraphDataset<SampleType>, SampleType> 
{
public:
    GraphDataset(std::vector<SampleType> samples_) : samples(std::move(samples_))
    {
    
    }

    virtual ~GraphDataset() override = default;

    virtual torch::optional<size_t> size() const override
    {
        return samples.size();
    }

    virtual SampleType get(size_t idx) override
    {
        return samples.at(idx);
    }

    virtual const std::vector<SampleType>& get_samples() const 
    { 
        return samples; 
    }

    virtual std::vector<SampleType>& get_samples() 
    { 
        return samples; 
    }

    virtual GraphDataset& to(torch::ScalarType Dtype, torch::ScalarType Itype) 
    {
        for(auto& sample : samples)
        {
            sample.to(Dtype, Itype);
        }

        return *this;
    }

protected:
    std::vector<SampleType> samples;
};
