-- job_submit.lua
-- This script is called by Slurm when a job is submitted.
-- It checks if the job was submitted without a script (likely salloc or srun)
-- and, if so, moves it from the gpu_long partition to the gpu_interac partition.

-- Number of shards per GPU (e.g., 8 means GPU is divided into 8 shards)
local NUM_SHARDS = 4

-- Patterns to match GPU slice requests in different formats
-- Handles: gres/gpu:type.2:1, gpu:type.2:1, type.2:1
local GPU_SLICE_PATTERNS = {
    "gres/gpu:([%w_]+)%.(%d+):(%d+)",  -- gres/gpu:type.2:1
    "gpu:([%w_]+)%.(%d+):(%d+)",        -- gpu:type.2:1
    "([%w_]+)%.(%d+):(%d+)"             -- type.2:1
}

-- Validate that slice requests have count = 1 (no multiple slices allowed)
-- Returns error message if invalid, nil if valid
local function validate_gpu_slice_count(str, field_name)
    slurm.log_info("validate_gpu_slice_count: " .. str .. " " .. field_name)
    if str == nil or str == "" then
        return nil
    end
    -- Pattern matches <gpu_name>.<denominator>:<count> format
    -- Handles formats like: gres/gpu:type.2:2, gpu:type.2:2, type.2:2
    local found_error = false
    local error_details = ""
    
    for _, pattern in ipairs(GPU_SLICE_PATTERNS) do
        string.gsub(str, pattern, function(gpu_name, denominator, count)
            slurm.log_info("validate_gpu_slice_count match: " .. gpu_name .. " " .. denominator .. " " .. count)
            local denom = tonumber(denominator)
            local cnt = tonumber(count)
            
            -- Check if denominator is 1 (full GPU - should use full GPU request instead)
            if denom and denom == 1 then
                found_error = true
                error_details = "ERROR: Slice size cannot be 1 (full GPU)\n"
                    .. "Field: " .. field_name .. "\n"
                    .. "Requested: " .. gpu_name .. "." .. denominator .. ":" .. count .. "\n"
                    .. "For a full GPU, use: gpu:" .. gpu_name .. ":" .. (cnt or 1) .. "\n"
                    .. "The largest slice size is 1/2 of a GPU (denominator must be >= 2)\n"
                return
            end
            
            -- Check if denominator exceeds NUM_SHARDS
            if denom and denom > NUM_SHARDS then
                found_error = true
                error_details = "ERROR: Requested slice size is too small\n"
                    .. "Field: " .. field_name .. "\n"
                    .. "Requested: " .. gpu_name .. "." .. denominator .. ":" .. count .. "\n"
                    .. "The smallest slice size is 1/" .. NUM_SHARDS .. " of a GPU\n"
                    .. "Maximum allowed denominator is " .. NUM_SHARDS .. "\n"
                return
            end
            
            -- Check if count > 1 (multiple slices not allowed)
            if cnt and cnt > 1 then
                found_error = true
                error_details = "ERROR: Multiple GPU slices of the same size are not allowed\n"
                    .. "Field: " .. field_name .. "\n"
                    .. "Requested: " .. gpu_name .. "." .. denominator .. ":" .. count .. "\n"
                    .. "You can only request:\n"
                    .. "  - A single slice with count=1 (e.g., " .. gpu_name .. "." .. denominator .. ":1)\n"
                    .. "  - Multiple full GPUs (e.g., gpu:" .. gpu_name .. ":" .. cnt .. ")\n"
            end
        end)
        if found_error then
            break
        end
    end
    if found_error then
        return error_details
    end
    return nil
end

-- Translate gpu.denominator:count pattern to gres/shard:gpu:calculated_count format
-- Example: type.2:1 (1/2 GPU) with NUM_SHARDS=8 becomes gres/shard:type:4 (4/8 = 1/2)
-- Handles formats like: gres/gpu:type.2:1, gpu:type.2:1, type.2:1
local function translate_gpu_shard(str)
    slurm.log_info("translate_gpu_shard: " .. str)
    if str == nil or str == "" then
        return str
    end
    -- Pattern matches <gpu_name>.<denominator>:<count> where:
    -- gpu_name is alphanumeric/underscore
    -- denominator is one or more digits
    -- count is one or more digits
    -- Handles optional "gres/" and "gpu:" prefixes
    local result = str
    
    for _, pattern in ipairs(GPU_SLICE_PATTERNS) do
        result = string.gsub(result, pattern, function(gpu_name, denominator, count)
            slurm.log_info("translate_gpu_shard match: " .. gpu_name .. " " .. denominator .. " " .. count)
            local denom = tonumber(denominator)
            local cnt = tonumber(count)
            -- Reject denominator = 1 (should use full GPU request) and ensure denominator is between 2 and NUM_SHARDS
            if denom and cnt and denom >= 2 and denom <= NUM_SHARDS then
                -- Calculate shard count: (count / denominator) * NUM_SHARDS
                local shard_count = math.floor((cnt / denom) * NUM_SHARDS + 0.5) -- Round to nearest integer
                -- Ensure shard_count is at least 1 if the request is valid
                if shard_count < 1 then
                    shard_count = 1
                end
                return "gres/shard:" .. gpu_name .. ":" .. shard_count
            end
            -- Return original if calculation fails or denominator is invalid
            return gpu_name .. "." .. denominator .. ":" .. count
        end)
    end
    return result
end

function slurm_job_submit(job_desc, part_list, submit_uid)
    slurm.log_info("job_submit.lua: Checking job submission type.")

    -- Extract gres and tres fields for shard translation
    local gres_requested = job_desc.gres or ""
    local tres_per_job = job_desc.tres_per_job or ""
    local tres_per_node = job_desc.tres_per_node or ""
    local tres_per_socket = job_desc.tres_per_socket or ""
    local tres_per_task = job_desc.tres_per_task or ""

    -- Validate slice requests (count must be 1)
    local error_msg = validate_gpu_slice_count(gres_requested, "gres")
        or validate_gpu_slice_count(tres_per_job, "tres_per_job")
        or validate_gpu_slice_count(tres_per_node, "tres_per_node")
        or validate_gpu_slice_count(tres_per_socket, "tres_per_socket")
        or validate_gpu_slice_count(tres_per_task, "tres_per_task")
    
    if error_msg then
        slurm.log_user('\n\027[00;31m========================================\n'
            .. error_msg
            .. '========================================\027[00m\n')
        return slurm.ERROR
    end

    -- Translate <gpu>.# patterns to gres/shard:<gpu>:# format
    gres_requested = translate_gpu_shard(gres_requested)
    tres_per_job = translate_gpu_shard(tres_per_job)
    tres_per_node = translate_gpu_shard(tres_per_node)
    tres_per_socket = translate_gpu_shard(tres_per_socket)
    tres_per_task = translate_gpu_shard(tres_per_task)

    -- Assign translated values back to job_desc
    if gres_requested ~= "" then
        job_desc.gres = gres_requested
    end
    if tres_per_job ~= "" then
        job_desc.tres_per_job = tres_per_job
    end
    if tres_per_node ~= "" then
        job_desc.tres_per_node = tres_per_node
    end
    if tres_per_socket ~= "" then
        job_desc.tres_per_socket = tres_per_socket
    end
    if tres_per_task ~= "" then
        job_desc.tres_per_task = tres_per_task
    end
    
    -- If gres is not set but one of the tres variables was translated to shard, set gres
    if (gres_requested == nil or gres_requested == "") then
        -- Check each tres variable for shard format
        local tres_vars = {tres_per_job, tres_per_node, tres_per_socket, tres_per_task}
        for _, tres_val in ipairs(tres_vars) do
            if tres_val and tres_val ~= "" then
                -- Check if it contains gres/shard: pattern and extract the full specification
                -- Pattern matches gres/shard:gpu_name:count (may be part of a larger string)
                local full_shard = tres_val:match("gres/shard:[%w_]+:%d+")
                if full_shard then
                    job_desc.gres = full_shard
                    slurm.log_info("Auto-set gres from tres variable: " .. full_shard)
                    break
                end
            end
        end
    end
 
    return slurm.SUCCESS;
end

function slurm_job_modify(job_id, job_desc, job_rec, part_list, modify_uid)
    -- This function is required by the job_submit/lua plugin,
    -- but we don't need it to do anything for this specific logic.
    return slurm.SUCCESS
end