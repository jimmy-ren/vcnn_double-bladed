function out_forward()
    global config mem;
    mem.activations{length(mem.activations)} = config.OUT_ACT(mem.activations{length(mem.activations)});
    config.GEN_OUTPUT();
    config.cost = config.COST_FUN(mem.output, mem.GT_output, config.batch_size);
end

