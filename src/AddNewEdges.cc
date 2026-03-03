#include "AddNewEdges.hh"

GraphSample AddNewEdges(const std::vector<PreProcessing::Hit>& hits,
                        const PreProcessing::PreprocessingParams& params,
                        GraphSample graph,
                        bool train,
                        torch::ScalarType Dtype, torch::ScalarType Itype)
{
    auto answer = graph.answer >= params.threshold;
    auto index = torch::squeeze(torch::argwhere(torch::squeeze(answer)));

    graph.edge_index = torch::index_select(graph.edge_index, 1, index);
    graph.edge_attr = torch::index_select(graph.edge_attr, 0, index);
    graph.answer = torch::index_select(graph.answer, 0, index);

    auto tracks = PostProcessing::GetTracks(graph, -1);

    std::vector<std::set<int>> row_outer_hits(params.num_rows), row_inner_hits(params.num_rows);
    for (size_t track_id = 0; track_id < tracks.size(); ++track_id)
    {
        std::map<int, std::set<int>> row_hit_ids;
        for (int i : tracks[track_id])
        {
            row_hit_ids[hits[i - 1].row_id].emplace(i);
        }
        auto it = row_hit_ids.begin();
        row_inner_hits[(*it).first].merge((*it).second);
        it = row_hit_ids.end();
        it--;
        row_outer_hits[(*it).first].merge((*it).second);
    }
    auto node_hit_id_flat = graph.node_hit_id.squeeze();
    auto node_hit_id_a = node_hit_id_flat.accessor<int, 1>();
    std::map<int, int> hit_index_lookup;
    for (int i = 0; i < node_hit_id_flat.size(0); ++i)
    {
        hit_index_lookup[node_hit_id_a[i]] = i;
    }

    std::vector<torch::Tensor> new_edge_indices, new_edge_features, new_edge_labels;

    for (size_t outer_row_id = 1; outer_row_id < params.num_rows - 3; ++outer_row_id)
    {
        for (size_t inner_row_id = outer_row_id + 2; inner_row_id < params.num_rows - 1; ++inner_row_id)
        {
            for (auto outer_hit_id : row_outer_hits[outer_row_id])
            {
                for (auto inner_hit_id : row_inner_hits[inner_row_id])
                {
                    float dphi = PreProcessing::CalcDphi(hits[outer_hit_id - 1].phi,
                                                         hits[inner_hit_id - 1].phi);
                    float dz = hits[inner_hit_id - 1].z - hits[outer_hit_id - 1].z;
                    float distance = std::sqrt(hits[outer_hit_id - 1].r *
                    hits[outer_hit_id - 1].r + hits[inner_hit_id - 1].r *
                    hits[inner_hit_id - 1].r - 2 * hits[outer_hit_id - 1].r *
                    hits[inner_hit_id - 1].r * std::cos(dphi) + dz * dz);
                    if (distance < params.d_max_2)
                    {
                        new_edge_indices.push_back(torch::tensor({hit_index_lookup[outer_hit_id],
                            hit_index_lookup[inner_hit_id]},
                            torch::TensorOptions().dtype(torch::kInt32)));

                        auto node_pos_1_tensor = graph.node_attr[hit_index_lookup[outer_hit_id]];
                        auto node_pos_2_tensor = graph.node_attr[hit_index_lookup[inner_hit_id]];
                        auto node_pos_1_a = node_pos_1_tensor.accessor<float, 1>();
                        auto node_pos_2_a = node_pos_2_tensor.accessor<float, 1>();
                        std::vector<float> node_pos_1_vector(params.node_attr_size), node_pos_2_vector(params.node_attr_size);
                        for (size_t k = 0; k < params.node_attr_size; ++k)
                        {
                            node_pos_1_vector[k] = node_pos_1_a[k];
                            node_pos_2_vector[k] = node_pos_2_a[k];
                        }
                        auto edge_feat = PreProcessing::GetEdgeFeatures(node_pos_1_vector, node_pos_2_vector, torch::kFloat32);
                        new_edge_features.push_back(edge_feat);

                        if (train)
                        {
                            int label = (hits[outer_hit_id - 1].track_id == hits[inner_hit_id - 1].track_id &&
                            hits[outer_hit_id - 1].track_id != -1 && hits[outer_hit_id - 1].pt >= params.pt_min) ? 1 : 0;
                            new_edge_labels.push_back(torch::tensor({static_cast<float>(label)},
                                                                    torch::TensorOptions().dtype(torch::kFloat32)));
                        }
                    }
                }
            }
        }
    }

    torch::Tensor new_edge_index = torch::stack(new_edge_indices, 0).transpose(0, 1);
    torch::Tensor new_edge_attr = torch::stack(new_edge_features, 0);

    torch::Tensor new_edge_start = torch::tensor({{graph.edge_attr.size(0)}}, torch::TensorOptions().dtype(torch::kInt32));

    graph.edge_index = torch::cat({graph.edge_index, new_edge_index}, 1);
    graph.edge_attr = torch::cat({graph.edge_attr, new_edge_attr}, 0);
    if (train)
    {
        torch::Tensor new_answer = torch::stack(new_edge_labels, 0);
        graph.answer = torch::cat({graph.answer, new_answer}, 0);
    }

    graph.new_edge_start = new_edge_start;

    return graph;
}
