#pragma once

#include <vector>

#include "GraphDataset.hh"
#include "PostProcessing.hh"
#include "PreProcessing.hh"

GraphSample AddNewEdges(const std::vector<PreProcessing::Hit>& hits,
                        const PreProcessing::PreprocessingParams& params,
                        GraphSample graph,
                        bool train = false,
                        torch::ScalarType Dtype = torch::kFloat32, torch::ScalarType Itype = torch::kInt32);
