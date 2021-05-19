## Program to run PPSeq on a section of data

# INPUTS:

# PPSeq_file = the PPSeq.jl file to use
# Data_directory = where to find the data to use
# Time = how long the segment is MUST BE FLOAT
# Neurons = how many neurons being used MUST BE INTEGER

if size(ARGS) != (4,)
    error("Not correct number of command line arguments")
end

PPSeq_file = ARGS[1]
Data_directory = ARGS[2]
max_time = parse(Float64, ARGS[3])
num_neurons = parse(Int64, ARGS[4])

# Import PPSeq
include(PPSeq_file)
const seq = PPSeq

# Other Imports
import DelimitedFiles: readdlm
import Random
import StatsBase: quantile
using CSV, DataFrames, Dates

function save_results(results, run_id)
   
    # ---------- anneal_latent_event_hist -----------
    latent_event_hist = results[:anneal_latent_event_hist]
    init_array = zeros(Float64, 0, 5)
    for (i, s) in enumerate(latent_event_hist)
        aux_array = zeros(Float64, length(s)+1, 5)
        for (j, z) in enumerate(s)
            aux_array[j, 1] = z.assignment_id
            aux_array[j, 2] = z.timestamp
            aux_array[j, 3] = z.seq_type
            aux_array[j, 4] = z.seq_warp
            aux_array[j, 5] = z.amplitude
        end
        aux_array[length(s)+1, :] = -1*ones(5)
        init_array = vcat(init_array, aux_array)
    end
    anneal_latent_event_hist_array = init_array
    ANNEAL_LATENT_EVENT_SHAPE = size(init_array)
   
    # ----------- latent_event_hist ------------
    latent_event_hist = results[:latent_event_hist]
    init_array = zeros(Float64, 0, 5)
    for (i, s) in enumerate(latent_event_hist)
        aux_array = zeros(Float64, length(s)+1, 5)
        for (j, z) in enumerate(s)
            aux_array[j, 1] = z.assignment_id
            aux_array[j, 2] = z.timestamp
            aux_array[j, 3] = z.seq_type
            aux_array[j, 4] = z.seq_warp
            aux_array[j, 5] = z.amplitude
        end
        aux_array[length(s)+1, :] = -1*ones(5)
        init_array = vcat(init_array, aux_array)
    end
    latent_event_hist_array = init_array
    LATENT_EVENT_SHAPE = size(init_array)
   
    latent_event_hist_frame = DataFrame(vcat(anneal_latent_event_hist_array, latent_event_hist_array),
        Symbol.(["assignment_id", "timestamp", "seq_type", "seq_warp", "amplitude"]))
   
    # --------- global_hist ----------
    global_vars_list = results[:globals_hist]
    N = size(global_vars_list[1].neuron_response_log_proportions)[1]
    R = size(global_vars_list[1].neuron_response_log_proportions)[2]
    init_array = zeros(Float64, 0, 3*R)
    seq_type_log_proportions = zeros(Float64, length(global_vars_list), R)
    bkgd_log_proportions = zeros(Float64, length(global_vars_list), N)
    for (i, s) in enumerate(global_vars_list)
        neuron_response = hcat(s.neuron_response_log_proportions, s.neuron_response_offsets,
        s.neuron_response_widths)
        init_array = vcat(init_array, neuron_response)
        seq_type_log_proportions[i, :] = s.seq_type_log_proportions
        bkgd_log_proportions[i, :] = s.bkgd_log_proportions
    end
    neuron_response_array = init_array
    seq_type_log_proportions_array = seq_type_log_proportions
    bkgd_log_proportions_array = bkgd_log_proportions
   
    # --------- anneal_global_hist ----------
        global_vars_list = results[:anneal_globals_hist]
    N = size(global_vars_list[1].neuron_response_log_proportions)[1]
    R = size(global_vars_list[1].neuron_response_log_proportions)[2]
    init_array = zeros(Float64, 0, 3*R)
    seq_type_log_proportions = zeros(Float64, length(global_vars_list), R)
    bkgd_log_proportions = zeros(Float64, length(global_vars_list), N)
    for (i, s) in enumerate(global_vars_list)
        neuron_response = hcat(s.neuron_response_log_proportions, s.neuron_response_offsets,
        s.neuron_response_widths)
        init_array = vcat(init_array, neuron_response)
        seq_type_log_proportions[i, :] = s.seq_type_log_proportions
        bkgd_log_proportions[i, :] = s.bkgd_log_proportions
    end
    anneal_neuron_response_array = init_array
    anneal_seq_type_log_proportions_array = seq_type_log_proportions
    anneal_bkgd_log_proportions_array = bkgd_log_proportions
   
    neuron_response_frame = DataFrame(vcat(anneal_neuron_response_array, neuron_response_array))
    seq_type_log_proportions_frame = DataFrame(vcat(anneal_seq_type_log_proportions_array, seq_type_log_proportions_array))
    bkgd_log_proportions_frame = DataFrame(vcat(anneal_bkgd_log_proportions_array, bkgd_log_proportions_array))
   
    # --------- assignment_hist ----------
    assigment_hist_frame = DataFrame(hcat(results[:anneal_assignment_hist], results[:assignment_hist]))
   
    aux_array = zeros(Float64, length(results[:log_p_hist]), 1)
    aux_array[:, 1] = results[:log_p_hist]
    log_p_hist_array = aux_array
   
    aux_array = zeros(Float64, length(results[:anneal_log_p_hist]), 1)
    aux_array[:, 1] = results[:anneal_log_p_hist]
    anneal_log_p_hist_array = aux_array
    log_p_hist_frame = DataFrame(vcat(anneal_log_p_hist_array, log_p_hist_array))
    # return log_p_hist_frame
   
    aux_array = zeros(Float64, length(results[:initial_assignments]), 1)
    aux_array[:, 1] = results[:initial_assignments]
    initial_assignments_frame = DataFrame(aux_array)
   
    results_directory = run_id*"_results/"
    if !isdir(results_directory)
        mkdir(results_directory)
    end
   
    CSV.write(results_directory*"latent_event_hist.csv", latent_event_hist_frame, writeheader=true)
    CSV.write(results_directory*"neuron_response.csv", neuron_response_frame, writeheader=true)
    CSV.write(results_directory*"seq_type_log_proportions.csv", seq_type_log_proportions_frame, writeheader=true)
    CSV.write(results_directory*"bkgd_log_proportions_array.csv", bkgd_log_proportions_frame, writeheader=true)
    CSV.write(results_directory*"log_p_hist.csv", log_p_hist_frame, writeheader=true)
    CSV.write(results_directory*"initial_assignments.csv", latent_event_hist_frame, writeheader=true)
   
    open(results_directory*"data_readme.txt", "w") do io
        write(io, "\n")
        write(io, "Latent_events: [N_spikes, iter] \n")
        write(io, "anneal_latent_event: "*string(ANNEAL_LATENT_EVENT_SHAPE)*"\n")
        write(io, "after_anneal_latent_event: "*string(LATENT_EVENT_SHAPE)*"\n")
        write(io, "full_latent_event: "*string(size(latent_event_hist_frame))*"\n")
        write(io, "Iterations separated by a row of -1 \n")
        write(io, "\n")
        write(io, "Neuron response: [log_proportions, offsets, widths] (neurons*iters) × 3R \n")
        write(io, "anneal_neuron_response: "*string(size(anneal_neuron_response_array))*"\n")
        write(io, "after_neuron_response: "*string(size(neuron_response_array))*"\n")
        write(io, "full_neuron_response: "*string(size(neuron_response_frame))*"\n")
        write(io, "\n")
        write(io, "assigment_hist: n_spikes x (100 iters anneal + 200 iters) \n")
        write(io, "assigment_hist: "*string(size(assigment_hist_frame))*"\n")
        write(io, "log_p_hist: "*string(size(log_p_hist_frame))*", (iters x 1)\n")
        write(io, "bkgd_log_proportions: "*string(size(bkgd_log_proportions_frame))*"\n")
        write(io, "seq_type_log_proportions: "*string(size(seq_type_log_proportions_frame))*"\n")
        write(io, "(anneal iters) + (without anneal iters) \n")
        write(io, "\n")
        write(io, "initial_assignments, "*string(size(initial_assignments_frame))*"\n")
        write(io, "For more details on globals and events, see \n https://github.com/rodrigcd/PPSeq.jl/blob/84d8c1da3f7fe55f93555ae94242d1309c7d44fb/src/model/structs.jl")
    end
   
    return latent_event_hist_frame
   
end

# Load spikes.
spikes = seq.Spike[]
file_name = Data_directory
for (n, t) in eachrow(readdlm(file_name, '\t', Float64, '\n'))
    push!(spikes, seq.Spike(Int(n), t))
end

# Setup the config details
config = Dict(

    # Model hyperparameters
    :num_sequence_types =>  5,
    :seq_type_conc_param => 1.0,
    :seq_event_rate => 1.0,

    :mean_event_amplitude => 100.0,
    :var_event_amplitude => 1000.0,
    
    :neuron_response_conc_param => 0.1,
    :neuron_offset_pseudo_obs => 1.0,
    :neuron_width_pseudo_obs => 1.0,
    :neuron_width_prior => 0.5,
    
    :num_warp_values => 1,
    :max_warp => 1.0,
    :warp_variance => 1.0,

    :mean_bkgd_spike_rate => 30.0,
    :var_bkgd_spike_rate => 30.0,
    :bkgd_spikes_conc_param => 0.3,
    :max_sequence_length => Inf,
    
    # MCMC Sampling parameters.
    :num_threads => 10,
    :num_anneals => 10,
    :samples_per_anneal => 100,
    :max_temperature => 40.0,
    :save_every_during_anneal => 10,
    :samples_after_anneal => 2000,
    :save_every_after_anneal => 10,
    :split_merge_moves_during_anneal => 10,
    :split_merge_moves_after_anneal => 10,
    :split_merge_window => 1.0,

);

# Then train the PPSeq Model
# Initialize all spikes to background process.
init_assignments = fill(-1, length(spikes))

# Construct model struct (PPSeq instance).
model = seq.construct_model(config, max_time, num_neurons)

# Run Gibbs sampling with an initial annealing period.
results = seq.easy_sample!(model, spikes, init_assignments, config);

# Save the results
save_results(results, "../Simple"*Dates.format(Dates.now(),"HH:MM"))