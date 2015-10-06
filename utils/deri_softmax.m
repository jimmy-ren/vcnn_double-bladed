function deri = deri_softmax(deri_cost)
    global config;
    deri = deri_cost / config.batch_size;
end
