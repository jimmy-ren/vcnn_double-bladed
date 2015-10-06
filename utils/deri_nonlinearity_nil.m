function deri = deri_nonlinearity_nil(deri_cost)
    global config;
    deri = deri_cost / config.batch_size;
end

