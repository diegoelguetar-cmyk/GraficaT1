import ctypes
import os
from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL
from grafica.utils import load_pipeline

dt = .001


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
    

def calentura(state, dt = .1, steps=1000):
    """
    obtiene los mayores valores del arreglo de posiciones, para ver el tamaño de la figura en cuestión
    """

    s = (steps, 3)
    states = np.zeros(s)
    states[0] = state

    for i in range(1, steps):
        states[i] = rk4_step(state, dt)
        state = states[i]


    rmax =np.array([np.max(states[:, 0]), np.max(states[:, 1]), np.max(states[:, 2])])
    rmin = np.array([np.min(states[:, 0]), np.min(states[:, 1]), np.min(states[:, 2])])

    return rmax, rmin 


def parametrizar(state, rmax, rmin):
    """
    Función que por cada evento calculado, parametrizará los datos para graficar en pantalla. 
    Notar que se pide calcular los valores minimos y máximos fuera de la función, por eficiencia. 
    retorna punto en pantalla usando Normalized Devide coordinates
    """
    point  = 2* (state - rmin)/(rmax-rmin ) - 1
    return point

def det_VBO(list_points, point_vew):
    """
    Determona el vertex Buffer object, donde segun tipo de perfil de visualizacion, el sistema elije la lista a graficar. 
    """
    x = list_points[0]
    y = list_points[1]
    z = list_points[2]
    l = np.len(z)


    if (point_vew == "xy"): 
        VBO= np.zeros(2*l)
        for i in range(l):
            VBO[2*i] = x[i]
            VBO[2*i +1] = y[i]
        return VBO


    elif( point_vew == "xz"):
        VBO= np.zeros(2*l)
        for i in range(l):
            VBO[2*i] = x[i]
            VBO[2*i +1] = z[i]
        return VBO
    
    elif(point_vew =="yz"): 
        VBO= np.zeros(2*l)
        for i in range(l):
            VBO[2*i] = y[i]
            VBO[2*i +1] = z[i]
        return VBO

    
def construir_VAO(vbo):
    """
    A partir de VBO, ensamblamos el vertex array program
    """
    l = 3*np.len(vbo[0, :])
    VAO = np.zeros(l)


def oscilador(theta, j= None , delta = .1):
    val = np.cos(theta)
    theta += delta

    if j == None :
        j = 1
    
    if (val <= 1 and val > -np.sqrt(3/2) and (j == 3 or j == 1)): # cos(0), cos(120) = cos(30)
        j = 1
        return"xy", j
    if (val <= -np.sqrt(3/2)): # cos(30), cos(240) = cos(-60)
        j = 2
        return "xz", j
    if (val > -1/np.sqrt(2) and val < 1 and  (j==2 or j== 3)):
        j = 3
        return "yz", j
        
#medimos valores máximos
r = calentura(np.array(1., .1, .1), dt)
RANGOS = {
    "xz": (-25.0, 25.0, 0.0, 50.0),
    "xy": (-25.0, 25.0, -30.0, 30.0),
    "yz": (-30.0, 30.0, 0.0, 50.0),
}
#estamos listos para función principal 

def Rossler(width, height, particles, steps, dt, plano, range, point_vew="xy"):
    """
    Atractor de 
    función principal
    podemos usar más de una partícula 
    VARIABLES   
    range: rango de valores max y min del caso. arreglo de 6, rmax_vec, rmin_vec
    """
    def integrate_and_collect(state, range, num_steps=200, dt=.001):
        """
        Evolución dinámica del problema, retorna puntos normalizados
        """
        s = (num_steps, 3)
        list_states = np.zeros(s, dtype= np.float32)
        i = 1
        list_states[0] = state

        while (i< num_steps):
            new_state = rk4_step(state, dt)
            list_states[i] = new_state
            state = new_state
            i+=1

        list_point = parametrizar(list_states, range[:3], range[3:])

        return list_point

    #definimos ventana 
    win = pyglet.window.Window(width, height, ñtcaption=f"Rossler ({plano})")

    # --- Framebuffer de acumulación con textura float ---
    accum_tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, accum_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_R16F,
        width, height, 0,
        GL.GL_RED, GL.GL_FLOAT, None
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D, accum_tex, 0
    )
    GL.glClearColor(0.0, 0.0, 0.0, 0.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # --- Pipeline de puntos ---
    pipeline_points = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "point_fragment.glsl",
    )


    vao_points = GL.glGenVertexArrays(1)
    vbo_points = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao_points)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_points)
    pos_loc = GL.glGetAttribLocation(pipeline_points.id, "position")
    GL.glEnableVertexAttribArray(pos_loc)
    GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0,
                             ctypes.c_void_p(0))
    GL.glBindVertexArray(0)

    # -- Pipeline de visualización --
    pipeline_vis = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "visualization.glsl",
    )

    points = integrate_and_collect(np.array((1., .1, .1)), r)
    points_2d = det_VBO(points, point_vew)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_points)

    GL.glBufferData(
        GL.GL_ARRAY_BUFFER,
        points_2d.nbytes,
        points_2d,
        GL.GL_STREAM_DRAW
    )


    GL.glDrawArrays(
        GL.GL_POINTS,
        0,
        len(points_2d) // 2
    )
  

    GL.glBindVertexArray(0)
    def tick(frame_time):
        if paused:
            return

        points = integrate_and_collect(
            state,
            num_steps=steps,
            dt=dt
        )

        if len(points) == 0:
            return

        points_2d = det_VBO(points, point_view)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        GL.glViewport(0, 0, width, height)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

        pipeline_points.use()

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_points)

        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            points_2d.nbytes,
            points_2d,
            GL.GL_STREAM_DRAW
        )

        GL.glBindVertexArray(vao_points)

        GL.glDrawArrays(
            GL.GL_POINTS,
            0,
            len(points_2d) // 2
        )

        GL.glBindVertexArray(0)

        GL.glDisable(GL.GL_BLEND)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def limpiar():
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @win.event
    def on_draw():
        win.clear()
        GL.glViewport(0, 0, width, height)

        pipeline_vis.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, accum_tex)

        sampler_loc = GL.glGetUniformLocation(pipeline_vis.id, "accum_tex")
        if sampler_loc != -1:
            GL.glUniform1i(sampler_loc, 0)

        exposure_loc = GL.glGetUniformLocation(pipeline_vis.id, "exposure")
        if exposure_loc != -1:
            GL.glUniform1f(exposure_loc, exposure)

        gpu_quad.draw(GL.GL_TRIANGLES)

    print(f"Atractor de Lorenz, proyección {plano}")
    print("Controles:")
    print("  ESPACIO: pausar/reanudar")
    print("  R: reiniciar")
    print("  +/-: duplicar o dividir el paso de integración dt")
    print("  ARRIBA/ABAJO: ajustar exposición")
    print(f"  Trayectorias: {particles}, pasos por frame: {steps}, dt: {dt}")

    pyglet.clock.schedule_interval(tick, 1 / 60.0)
    pyglet.app.run()

