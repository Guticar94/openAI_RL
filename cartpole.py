# Llamar librerías
exec(open("libraries.py").read())

# Inicializar
env = gym.make('CartPole-v1')
states = env.observation_space.shape[0] # Número de estados posibles
actions = env.action_space.n # Número de acciones posibles
episodes = 10 # Número de simulaciones

# Correr las simulaciones
for episode in range(1, episodes+1): #Se resetean las variables globales para cada simulación
    state = env.reset() # Se resetea la matriz de inicio para cada simulación
    done = False # Rese resetea Booleano que determina el fin de la simulación
    score = 0 # Se resetea Puntaje del juego

    # Ciclo para cada simulación
    while not done: # Mientras el booleano sea Falso
        env.render() # Se inicia el ambiente
        action = env.action_space.sample() # Se realiza una accione de forma aleatoria
        n_state, reward, done, info = env.step(action) # Se capturan las variables globales para cada iteración
        score+= reward # Se actualiza el score
    #print('Episode:{} Score:{}'.format(episode,score))

# Llamar funciones
exec(open("functions.py").read())

# Correr el modelo de RL
model = bulid_model(states, actions) # Instanciar modelo DL
dqn = build_agent(model, actions) # Instanciar modelo RL
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Validar score del modelo
scores = dqn.test(env, nb_episodes = 5, visualize = False)
#print(np.mean(scores.history['episode_reward']))

# Salvar los pesos del modelo
dqn.save_weights('dqn.weights.h5f', overwrite=True)



