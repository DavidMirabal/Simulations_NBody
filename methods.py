class integrate:
    def __init__(self, y, f, args):
        self.x = y[0]
        self.v = y[1]
        self.f_x = f[0]
        self.f_v = f[1]
        self.args = args

    def euler(self, h):
        return (
            self.x + self.f_x(self.v) * h,
            self.v + self.f_v(self.x, self.args) * h,
        )

    def runge_kutta2(self, h):
        xmid = self.x + 0.5 * self.f_x(self.v) * h
        vmid = self.v + 0.5 * self.f_v(self.x, self.args) * h
        return (
            self.x + self.f_x(vmid) * h,
            self.v + self.f_v(xmid, self.args) * h,
        )

    def runge_kutta4(self, h):
        k1_x = self.f_x(self.v)
        k1_v = self.f_v(self.x, self.args)
        k2_x = self.f_x(self.v + k1_v * h * 0.5)
        k2_v = self.f_v(self.x + k2_x * h * 0.5, self.args)
        k3_x = self.f_x(self.v + k2_v * h * 0.5)
        k3_v = self.f_v(self.x + k3_x * h * 0.5, self.args)
        k4_x = self.f_x(self.v + k3_v * h)
        k4_v = self.f_v(self.x + k4_x * h, self.args)
        return self.x + h / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x), self.v + h / 6 * (
            k1_v + 2 * k2_v + 2 * k3_v + k4_v
        )
