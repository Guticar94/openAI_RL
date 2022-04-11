# Llamar librerías
exec(open("libraries.py").read())

# Llamar funciones
exec(open("functions.py").read())


# Instanciar variables
env = gym.make('CartPole-v1')
states = env.observation_space.shape[0] # Número de estados posibles
actions = env.action_space.n # Número de acciones posibles
model = bulid_model(states, actions) # Instanciar modelo DL
dqn = build_agent(model, actions) # Instanciar modelo RL
dqn.compile(Adam(lr=1e-3), metrics=['mae']) # Compilar modelo
dqn.load_weights('dqn.weights.h5f') # cargar pesos

# Testear modelo
dqn_ = dqn.test(env, nb_episodes = 5, visualize = True)
