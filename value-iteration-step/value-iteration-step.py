def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = [0.0] * num_states
    
    for s in range(num_states):
        max_q = float('-inf')
        num_actions = len(transitions[s])
        
        for a in range(num_actions):
            # 1. Compute the expected future value: sum(T(s,a,s') * V(s'))
            expected_future_value = 0.0
            for s_prime in range(num_states):
                expected_future_value += transitions[s][a][s_prime] * values[s_prime]
            
            # 2. Compute the Q-value for taking action a in state s
            q_value = rewards[s][a] + gamma * expected_future_value
            
            # 3. Keep track of the maximum Q-value found for this state
            if q_value > max_q:
                max_q = q_value
                
        # The new value for state s is the maximum Q-value
        new_values[s] = max_q
        
    return new_values