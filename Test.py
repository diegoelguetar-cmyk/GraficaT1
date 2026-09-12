import numpy as np 

def Rossler_step(state, a=0.2, b=0.2, c=5.7):
    """
    Retorna un estado del sistema en un tiempo dt dado un estado inicial state. 
    El sistema de Rössler es un sistema dinámico no lineal que exhibe comportamiento caótico.
    Los parámetros a, b y c controlan la dinámica del sistema.0    
    """

    x,y,z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)

    return np.array([dx, dy, dz])

def rk4_step(state, dt):

    k1 = Rossler_step(state)
    k2 = Rossler_step(state + k1*dt/2)
    k3 = Rossler_step(state + k2*dt/2)
    k4 = Rossler_step(state + k3*dt )

    new_state = state + (k1 + 2*k2 + 2*k3 + k4)*dt/6

    return new_state


def test_calentura(state, steps = 40000):
    """
    Simula el sistema de Rössler durante un número de pasos dado.
    Devuelve un array con los estados del sistema en cada paso.
    """
    original_state = state
    for dt in [0.1, 0.1, 0.01, 0.0001]:
        state = original_state
        x = True
        s = (steps, 3)
        states = np.zeros(s)
        states[0, :] = state

        for i in range(1, steps):
            states[i] = rk4_step(state, dt)
            state = states[i]

            if not np.all(np.isfinite(states)):
                print(f"{dt} exploted")
                print(f"{i}")
                x = False
                break
            
            
        if x:
            print(f"correct {dt}")

    return states

test_calentura(np.array((1, .1, .1)))

               