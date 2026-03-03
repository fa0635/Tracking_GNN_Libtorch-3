#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>

#include <yaml-cpp/yaml.h>

#include <torch/optim.h>
#include <torch/script.h>
#include <torch/nn/functional.h>

#include "EdgeClassificationGNN.hh"
#include "GraphDataLoader.hh"
#include "PreProcessing.hh"
#include "PostProcessing.hh"
#include "AddNewEdges.hh"

std::vector<GraphSample> load_graph_samples(const std::vector<std::string>& file_paths)
{
    std::vector<GraphSample> samples;

    for (const auto& path : file_paths)
    {
        if (std::filesystem::exists(path))
        {
            torch::jit::script::Module module = torch::jit::load(path);

            auto edge_index = module.attr("edge_index").toTensor();
            auto node_attr = module.attr("node_attr").toTensor();
            auto node_hit_id = module.attr("node_hit_id").toTensor();
            auto edge_attr = module.attr("edge_attr").toTensor();
            auto answer = module.attr("answer").toTensor();

            samples.emplace_back(edge_index, node_attr, node_hit_id, edge_attr, answer);
        }
        else
        {
            throw std::invalid_argument("load_graph_samples: File " + path + " is not found.");
        }
    }

    return samples;
}

void SelectHitsForTrainingFromCSV(const std::string& hit_file_name,
                                  const std::string& track_file_name,
                                  const std::string& truth_file_name,
                                  std::vector<PreProcessing::Hit>& hits)
{
    hits.clear();

    auto ParseCSVLine = [](const std::string& line) -> std::vector<std::string>
    {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ','))
        {
            tokens.push_back(token);
        }

        return tokens;
    };

    std::vector<int> hit_id_vec, row_id_vec, sector_id_vec;
    std::vector<float> z_vec, r_vec, phi_vec;

    std::ifstream hits_file(hit_file_name);
    if (!hits_file.is_open())
    {
        throw std::runtime_error("SelectHitsForTraining: Cannot open hit file: " + hit_file_name);
    }

    std::string line;
    bool first_line = true;

    while (std::getline(hits_file, line))
    {
        if (first_line)
        {
            first_line = false;
            continue;
        }

        auto tokens = ParseCSVLine(line);

        if (tokens.size() >= 6)
        {
            int hit_id = std::stoll(tokens[0]);
            float x = std::stod(tokens[1]);
            float y = std::stod(tokens[2]);
            float z = std::stod(tokens[3]);
            int sector_id = std::stoll(tokens[4]);
            int row_id = std::stoll(tokens[5]);
            float r = std::sqrt(x * x + y * y);
            float phi = std::atan2(y, x);

            hit_id_vec.push_back(hit_id);
            z_vec.push_back(z);
            r_vec.push_back(r);
            phi_vec.push_back(phi);
            row_id_vec.push_back(row_id);
            sector_id_vec.push_back(sector_id);
        }
    }

    hits_file.close();

    std::map<int, int> hit_to_track;
    std::map<int, float> track_to_pt;

    std::ifstream truth_file(truth_file_name);

    if (!truth_file.is_open())
    {
        throw std::runtime_error("SelectHitsForTraining: Cannot open truth file: " + truth_file_name);
    }

    first_line = true;

    while (std::getline(truth_file, line))
    {
        if (first_line)
        {
            first_line = false;
            continue;
        }

        auto tokens = ParseCSVLine(line);

        if (tokens.size() >= 2)
        {
            int hit_id = std::stoll(tokens[0]);
            int track_id = std::stoll(tokens[1]);

            hit_to_track[hit_id] = track_id;
        }
    }

    truth_file.close();

    std::ifstream tracks_file(track_file_name);
    if (!tracks_file.is_open())
    {
        throw std::runtime_error("SelectHitsForTraining: Cannot open track file: " + track_file_name);
    }

    first_line = true;

    while (std::getline(tracks_file, line))
    {
        if (first_line)
        {
            first_line = false;
            continue;
        }

        auto tokens = ParseCSVLine(line);

        if (tokens.size() >= 3)
        {
            int track_id = std::stoll(tokens[0]);
            float pt = std::stod(tokens[2]);

            track_to_pt[track_id] = pt;
        }
    }

    tracks_file.close();

    size_t n_hits = hit_id_vec.size();
    hits.reserve(n_hits);

    for (size_t i = 0; i < n_hits; ++i)
    {
        PreProcessing::Hit hit;
        hit.hit_id = hit_id_vec[i];
        hit.z = z_vec[i];
        hit.r = r_vec[i];
        hit.phi = phi_vec[i];
        hit.row_id = row_id_vec[i];
        hit.sector_id = sector_id_vec[i];

        auto track_it = hit_to_track.find(hit.hit_id);

        if (track_it != hit_to_track.end())
        {
            hit.track_id = track_it->second;

            auto pt_it = track_to_pt.find(hit.track_id);

            if (pt_it != track_to_pt.end())
            {
                hit.pt = pt_it->second;
            }
            else
            {
                hit.pt = 0.0;
            }
        }
        else
        {
            hit.track_id = -1;
            hit.pt = 0.0;
        }

        hits.push_back(hit);
    }
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::steady_clock::now();

    PreProcessing::PreprocessingParams params;

    try
    {
        params = PreProcessing::LoadConfig("../configs/preprocessing_parameters.yaml");
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error when reading the config file: " << ex.what() << '\n';
        std::exit(1);
    }

    std::cout << "Starting processing of " << (params.dataset_size * params.n_phi_sections * params.n_eta_sections) << " files...\n";

    std::vector<std::string> test_file_paths;
    for (size_t evtid = 0; evtid < params.dataset_size; ++evtid)
    {
        for (size_t section_id = 0; section_id < params.n_phi_sections * params.n_eta_sections; ++section_id)
        {
            std::ostringstream oss;
            oss << params.output_dir
            << "event_" << evtid
            << "_section_" << section_id
            << "_graph.pt";

            test_file_paths.push_back(oss.str());
        }
    }

    std::vector<std::string> output_file_paths;
    for (size_t evtid = 0; evtid < params.dataset_size; ++evtid)
    {
        for (size_t section_id = 0; section_id < params.n_phi_sections * params.n_eta_sections; ++section_id)
        {
            std::ostringstream oss;
            oss << params.graph_dir_2
            << "event_" << evtid
            << "_section_" << section_id
            << "_graph.pt";

            output_file_paths.push_back(oss.str());
        }
    }

    std::vector<GraphSample> test_data;

    try
    {
        test_data = load_graph_samples(test_file_paths);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        std::exit(1);
    }

    auto model = EdgeClassificationGNN(params.node_attr_size, params.edge_attr_size);
    //model->to(torch::kCUDA);

    try
    {
        model->load_model(params.saved_model_file);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        std::exit(1);
    }

    model->eval();
    torch::NoGradGuard no_grad;

    for (auto& graph : test_data)
    {
        //graph = graph.to(torch::kCUDA);

        auto edge_index = graph.edge_index;
        auto node_attr  = graph.node_attr;
        auto edge_attr  = graph.edge_attr;

        auto answer = model->forward(edge_index, node_attr, edge_attr);

        graph.answer = answer;
    }

    for (size_t evtid = 0; evtid < params.dataset_size; ++evtid)
    {
        try
        {
            std::string hit_file_name = params.input_dir + "event_" + std::to_string(evtid) + "_hits.csv";
            std::string truth_file_name = params.input_dir + "event_" + std::to_string(evtid) + "_truth.csv";
            std::string track_file_name = params.input_dir + "event_" + std::to_string(evtid) + "_tracks.csv";

            std::vector<PreProcessing::Hit> hits;

            SelectHitsForTrainingFromCSV(hit_file_name, track_file_name, truth_file_name, hits);

            for (size_t section_id = 0; section_id < params.n_phi_sections * params.n_eta_sections; ++section_id)
            {
                auto& graph = test_data[evtid * params.n_phi_sections * params.n_eta_sections + section_id];

                auto graph_2 = AddNewEdges(hits, params, graph, true);

                PreProcessing::SaveGraphSample(graph_2, output_file_paths[evtid * params.n_phi_sections * params.n_eta_sections + section_id]);
            }
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Error processing event " << evtid << ": " << ex.what() << std::endl;
            std::exit(1);
        }
    }

    std::cout << "Processing completed.\n";

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Total CPU/GPU time: " << elapsed.count() << " s.\n";

    return 0;
}
