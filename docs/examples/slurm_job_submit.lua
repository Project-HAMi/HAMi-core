-- job_submit.lua for softmig (Digital Research Alliance Canada)
-- Routes jobs to appropriate partitions based on GPU slice type
-- Validates GPU slice requests (only 1 slice allowed for fractional GPUs)
-- Only full GPU partitions (gpubase_*) can request multiple GPUs
--
-- Add this to your existing job_submit.lua or replace the relevant sections

function slurm_job_submit(job_desc, part_list, submit_uid)
    -- Detect GRES type and route to appropriate partition family
    local gres_prefix = "gpubase_"
    local gres_requested = job_desc.gres or ""
    local is_fractional = false
   
    -- Validate GPU slice count - only allow 1 slice for ALL fractional GPUs
    if string.find(gres_requested, "l40s%.2") then
        -- Check if requesting more than 1 half GPU
        local count = gres_requested:match("l40s%.2:(%d+)")
        if count and tonumber(count) > 1 then
            slurm.log_user('\n\027[00;31m----------------------------------------\n'
                ..'ERROR: You cannot request more than 1 half GPU slice.\n'
                ..'Requested: gpu:l40s.2:' .. count .. '\n'
                ..'Maximum allowed: gpu:l40s.2:1\n'
                ..'If you need more GPU resources, please use full GPUs: --gres=gpu:l40s:' .. count .. '\n'
                ..'----------------------------------------\027[00m')
            return slurm.ERROR
        end
        gres_prefix = "gpuhalf_"
        is_fractional = true
        slurm.log_info("job_submit.lua: Detected half GPU slice request")
       
    elseif string.find(gres_requested, "l40s%.4") then
        -- Check if requesting more than 1 quarter GPU
        local count = gres_requested:match("l40s%.4:(%d+)")
        if count and tonumber(count) > 1 then
            slurm.log_user('\n\027[00;31m----------------------------------------\n'
                ..'ERROR: You cannot request more than 1 quarter GPU slice.\n'
                ..'Requested: gpu:l40s.4:' .. count .. '\n'
                ..'Maximum allowed: gpu:l40s.4:1\n'
                ..'If you need more GPU resources, please use half GPUs: --gres=gpu:l40s.2:1\n'
                ..'or full GPUs: --gres=gpu:l40s:' .. math.ceil(tonumber(count)/4) .. '\n'
                ..'----------------------------------------\027[00m')
            return slurm.ERROR
        end
        gres_prefix = "gpuquarter_"
        is_fractional = true
        slurm.log_info("job_submit.lua: Detected quarter GPU slice request")
        
    elseif string.find(gres_requested, "l40s%.8") then
        -- Check if requesting more than 1 eighth GPU
        local count = gres_requested:match("l40s%.8:(%d+)")
        if count and tonumber(count) > 1 then
            slurm.log_user('\n\027[00;31m----------------------------------------\n'
                ..'ERROR: You cannot request more than 1 eighth GPU slice.\n'
                ..'Requested: gpu:l40s.8:' .. count .. '\n'
                ..'Maximum allowed: gpu:l40s.8:1\n'
                ..'If you need more GPU resources, please use quarter GPUs: --gres=gpu:l40s.4:1\n'
                ..'half GPUs: --gres=gpu:l40s.2:1, or full GPUs: --gres=gpu:l40s:' .. math.ceil(tonumber(count)/8) .. '\n'
                ..'----------------------------------------\027[00m')
            return slurm.ERROR
        end
        gres_prefix = "gpueighth_"
        is_fractional = true
        slurm.log_info("job_submit.lua: Detected eighth GPU slice request")
       
    elseif string.find(gres_requested, "l40s%.1") or string.find(gres_requested, "l40s:") or string.find(gres_requested, "gpu:") then
        gres_prefix = "gpubase_"
        slurm.log_info("job_submit.lua: Detected full GPU request")
    end
    
    -- Additional validation: Check if fractional GPU partition is trying to request multiple GPUs
    -- This catches cases where partition is explicitly set but GRES requests multiple
    if is_fractional then
        -- Extract GPU count from any GRES format
        local gpu_count = 1
        if string.find(gres_requested, "l40s%.2:") then
            gpu_count = tonumber(gres_requested:match("l40s%.2:(%d+)")) or 1
        elseif string.find(gres_requested, "l40s%.4:") then
            gpu_count = tonumber(gres_requested:match("l40s%.4:(%d+)")) or 1
        elseif string.find(gres_requested, "l40s%.8:") then
            gpu_count = tonumber(gres_requested:match("l40s%.8:(%d+)")) or 1
        elseif string.find(gres_requested, "l40s:") then
            gpu_count = tonumber(gres_requested:match("l40s:(%d+)")) or 1
        elseif string.find(gres_requested, "gpu:") then
            gpu_count = tonumber(gres_requested:match("gpu:(%d+)")) or 1
        end
        
        if gpu_count > 1 then
            slurm.log_user('\n\027[00;31m----------------------------------------\n'
                ..'ERROR: Fractional GPU slices (l40s.2, l40s.4, l40s.8) cannot request multiple GPUs.\n'
                ..'Requested: ' .. gres_requested .. '\n'
                ..'Only full GPU partitions (gpubase_*) allow multiple GPUs.\n'
                ..'Please use: --gres=gpu:l40s:' .. gpu_count .. ' for multiple GPUs.\n'
                ..'----------------------------------------\027[00m')
            return slurm.ERROR
        end
    end

    if job_desc.partition == "gpu_long" then
       job_desc.partition = gres_prefix .. "bygpu_b5"
    end

    if job_desc.time_limit == nil or job_desc.time_limit >= 4294967293 then
       job_desc.time_limit = 60
    end

    local cluster = 'vulcan'
    local queues = gres_prefix .. "bygpu_"

    local wallTimes = {180, 720, 1440, 4320, 10080, 40320}
    local longestBucket = 5
    local walltime = job_desc.time_limit
    local partition = job_desc.partition

    slurm.log_info("job_submit.lua: Current walltime: " .. walltime)

    local buc = 'b1'
    for i = #wallTimes, 2, -1 do
        if walltime > wallTimes[i - 1] then
            buc = 'b' .. i
            if i > longestBucket then
                wallTimeText = wallTimes[longestBucket] / 1440
                slurm.log_user('This job exceeds the maximum walltime of ' ..wallTimeText..' days on '..cluster..'.\n'
                ..'Please resubmit with a shorter walltime, or on a different cluster that allows longer walltimes.')
                return slurm.ERROR
            end
            break
        end
    end
    queues = queues .. buc
    slurm.log_info("job_submit.lua: Current estimated job partition: " .. queues)

    if job_desc.script == nil or job_desc.script == "" then
        slurm.log_info("job_submit.lua: No script detected (likely salloc/srun).")
        if job_desc.partition then
            slurm.log_info("job_submit.lua: Job is currently in partition " .. job_desc.partition)
        else
            slurm.log_info("job_submit.lua: Job currently has no partition set")
        end

        if walltime > 480 then
           slurm.log_user('This job exceeds the maximum walltime for interactive jobs on ' .. cluster ..'.\n'
              ..'Please resubmit with a walltime less than 8 hours.')
           return slurm.ERROR
        end

        if job_desc.partition == nil then
           job_desc.partition = gres_prefix .. "interac"
        else
           job_desc.partition = gres_prefix .. "interac"
        end
    else
        if job_desc.partition == nil then
           job_desc.partition = queues
        else
           buc = partition:sub(partition:len())
           if walltime <= wallTimes[tonumber(buc)] then
              return slurm.SUCCESS
           else
              slurm.log_user('\n\027[00;31m----------------------------------------\n'
                  ..'The specified partition does not exist, or the submitted job cannot fit in it...\n'
                  ..'Please specify a different partition, or simply submit the job without the --partition option,\n'
                  ..'the scheduler will redirect it to the most suitable partition automatically\n'
                  ..'----------------------------------------\027[00m')
              return slurm.ERROR
           end
        end
    end

    return slurm.SUCCESS;
end

function slurm_job_modify(job_id, job_desc, job_rec, part_list, modify_uid)
    return slurm.SUCCESS
end

