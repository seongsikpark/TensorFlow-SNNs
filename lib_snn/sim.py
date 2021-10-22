


# global clock - time step
class GLB_CLK():
    def __init__(self):
        GLB_CLK.t = 0

    def __call__(self):
        GLB_CLK.t += 1

    def reset(self):
        GLB_CLK.t = 0



glb_t = GLB_CLK()