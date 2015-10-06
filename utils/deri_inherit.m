function deri = deri_inherit(deri_cost)
    global config mem;
    deri = deri_cost .* config.DERI_NONLINEARITY(mem.output) / config.batch_size;
end
