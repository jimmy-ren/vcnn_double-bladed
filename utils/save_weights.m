function save_weights(path)
    global config;
    model = config;
    save(path, '-v7.3', 'model');
end
