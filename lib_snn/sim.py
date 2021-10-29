


# global clock - time step
class GLB_CLK():
    def __init__(self):
        GLB_CLK.t = 1

    def __call__(self):
        GLB_CLK.t += 1

    def reset(self):
        GLB_CLK.t = 1


# global configurations
class GLB():
    def __init__(self):
        GLB.model_compiled = False

    def model_compile_done(self):
        GLB.model_compiled = True







#
glb = GLB()
glb_t = GLB_CLK()