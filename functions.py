# Función modelo de DL
def bulid_model(states, actions): # Valores de entrada, matriz de estados, y posibles acciones
    model =  Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

# Función modelo RL
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length=1)
    dqn = DQNAgent(model = model, memory = memory, policy = policy,
                nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2)
    return dqn