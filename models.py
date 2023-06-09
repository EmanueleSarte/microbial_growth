import numpy as np
import matplotlib.pyplot as plt
import scipy


class GenericModel:
    def __init__(self, fixed_labels, var_labels, fixed_params=None, variable_params=None):
        self.fix_params_label = fixed_labels.copy()
        self.var_params_label = var_labels.copy()

        if fixed_params:
            self.fix_params = fixed_params.copy()

        if variable_params:
            self.var_params = variable_params.copy()
            self.var_params_initial = self.var_params.copy()

    def X(self, t):
        raise NotImplementedError()

    def S(self, t):
        raise NotImplementedError()

    def pdf(self, t, params=None):
        raise NotImplementedError()

    def log_pdf(self, params, *args):
        raise NotImplementedError()

    def log_lkl(self, params, *args):
        raise NotImplementedError()

    def log_prior(self, params, *args):
        raise NotImplementedError()

    def log_prob(self, params, spans, *args):
        raise NotImplementedError()

    def update_params(self, new_params):
        self.var_params = new_params.copy()

    def params_after_division(self, final_X):
        raise NotImplementedError()

    def find_best_tmax(self):
        test_range = np.arange(1, 40, 1)
        test_values = self.S(t=test_range)
        mask = np.argmax(np.isclose(test_values, 0, atol=1e-5, rtol=0), axis=0)
        max_t = test_range[mask]

        if mask == len(test_range) - 1:
            print("The maximum time range could be incorrect")
        return max_t

    def get_figure(self, ax=None):
        tmax = self.find_best_tmax()
        t = np.linspace(0, tmax, 1000)

        S = self.S(t)
        pdf = self.pdf(t)
        print(f"Best time range is [0, {tmax}]")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(t, S, label="S(t)")
        ax.plot(t, pdf, label="PDF")

        ax.set_xlabel("t (hours)")
        ax.legend()
        title = f"{self.__class__.__name__}  " + self.get_params_str()
        ax.set_title(title, wrap=True)
        plt.tight_layout()

    def get_params_str(self):
        params_list = [a for a in zip(self.fix_params_label, self.fix_params)]
        params_list += [a for a in zip(self.var_params_label, self.var_params_initial)]
        return " ".join([f"{k}={v:.3g}" for k, v in params_list])


class Model1_1(GenericModel):
    def __init__(self, m0=None, w1=None, w2=None, u=None, v=None):
        fixed_labels = ["w1", "w2", "u", "v"]
        var_labels = ["m0"]
        if (m0 is not None) and (w1 is not None) and (w2 is not None) and (u is not None) and (v is not None):
            self.m0 = m0
            self.w1 = w1
            self.w2 = w2
            self.u = u
            self.v = v
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels,
                             fixed_params=[w1, w2, u, v], variable_params=[m0])
        else:
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels)

    def X(self, t):
        value = (self.m0 + self.u) * np.exp(self.w1 * t) - self.u
        return value.reshape(-1, 1)

    def S(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

        term1 = w2 * (u/v - 1) * t
        term2 = w2 * ((m0 + u)/(w1*v)) * (np.exp(w1 * t) - 1)
        res = np.exp(term1 - term2)
        return res

    # def S_deriv(self, t):
    #     m0, w1, w2, u, v =     self.m0, self.w1, self.w2, self.u, self.v

    #     term1 = w2 * (u/v - 1) * t
    #     term2 = w2 * (m0 + u)/(w1*v) * (np.exp(w1 * t) - 1)
    #     res = np.exp(term1 - term2) * (w2 * (u/v - 1) - w2 * (m0 + u) * np.exp(w1*t) / v)
    #     return res

    def pdf(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        term1 = w2 * (u/v - 1) * t
        term2 = w2 * ((m0 + u)/(w1*v)) * (np.exp(w1 * t) - 1)
        h = w2 * (1 + (np.exp(w1 * t) * (m0 + u) - u)/v)
        out = (np.exp(term1 - term2)) * h
        # out[out == 0] = 1e-300
        return out

    def log_pdf(self, params, life_spans, m0s):
        # w1, w2, u, v = params
        # for t, m0 in zip(life_spans, m0s):

        w1, w2, u, v = params

        out = np.zeros(len(life_spans))
        for i, (t, m) in enumerate(zip(life_spans, m0s)):
            term1 = w2 * (u/v - 1) * t
            term2 = w2 * ((m + u)/(w1*v)) * (np.exp(w1 * t) - 1)
            h = w2 * (1 + (np.exp(w1 * t) * (m + u) - u)/v)

            value1 = term1 - term2
            # value2 = np.log((w2 * (u/v - 1) - w2 * (m + u) * np.exp(w1*t) / v))
            value2 = np.log(h)

            out[i] = value1 + value2

        return out

    def log_lkl(self, params, life_spans, m_zeros):
        log_pdfs = self.log_pdf(params, life_spans, m_zeros)
        if 0 in log_pdfs:
            print("AHHHH")
        # if np.any(np.isinf(log_pdfs)):
        #     print("XDZZ")
        #     return -np.inf

        return np.sum(log_pdfs)

    def log_prior(self, params):
        w1, w2, u, v = params
        if 0 < w1 <= 6 and 0 < w2 <= 6 and 0 < u <= 6 and 0 < v <= 30 and u < v:
            return 0.0
        else:
            return -np.inf

    def log_prob(self, params, spans, m_zeros):
        check = self.log_prior(params)
        if not np.isfinite(check):
            return -np.inf
        else:
            return check + self.log_lkl(params, spans, m_zeros)

    def params_after_division(self, final_X):
        return np.array([final_X[0] / 2])

    def update_params(self, new_params):
        super().update_params(new_params)
        self.m0 = new_params[0]


class Model1_2(GenericModel):
    def __init__(self, m0=None, w1=None, w2=None, u=None, v=None):
        fixed_labels = ["w1", "w2", "u", "v"]
        var_labels = ["m0"]
        if (m0 is not None) and (w1 is not None) and (w2 is not None) and (u is not None) and (v is not None):
            self.m0 = m0
            self.w1 = w1
            self.w2 = w2
            self.u = u
            self.v = v
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels,
                             fixed_params=[w1, w2, u, v], variable_params=[m0])
        else:
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels)

    def X(self, t):
        param1 = self.m0 * np.exp(self.w1 * t)
        return param1.reshape(-1, 1)

    def S(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

        res = np.ones(shape=t.shape)
        th = np.log(u / m0) / w1

        mask = t >= th
        piece = -w2 * m0 / ((u + v) * w1) * (np.exp(w1 * t) - np.exp(w1 * th) + w1 * v * (t - th) / m0)
        res[mask] = np.exp(piece)[mask]
        return res

        # if th > 0:
        #     t = t - th

        # factor = - (w2 * m0) / (w1 * (u + v))
        # term1 = np.exp(w1 * t) - 1
        # term2 = w1 * t * v / m0
        # res[mask] = np.exp(factor * (term1 + term2))[mask]
        # return res

    def pdf(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

        res = np.zeros(shape=t.shape)
        th = np.log(u / m0) / w1

        mask = t >= th
        piece = -w2 * m0 / ((u + v) * w1) * (np.exp(w1 * t) - np.exp(w1 * th) + w1 * v * (t - th) / m0)

        res[mask] = (-np.exp(piece) * (-w2 * m0 / (w1 * (u + v))) * (np.exp(w1 * t) * w1 + v * w1 / m0))[mask]
        return res

        # th = np.log(u / m0) / w1

        # mask = t >= th
        # if th > 0:
        #     t = t - th

        # factor = - (w2 * m0) / (w1 * (u + v))
        # term1 = np.exp(w1 * t) - 1
        # term2 = w1 * t * v / m0

        # h = w2 * (m0 * np.exp(w1 * t) + v) / (u + v)

        # res = np.zeros(shape=len(t))
        # res[mask] = (np.exp(factor * (term1 + term2)) * h)[mask]
        # return res

    def log_pdf(self, params, life_spans, m0s):
        w1, w2, u, v = params

        res = np.zeros(shape=len(life_spans))
        for i, (t, m) in enumerate(zip(life_spans, m0s)):
            # th = np.log(u / m) / w1
            # if t < th:
            #     res[i] = np.log(1e-4)
            #     # return res
            # else:
            #     t = t - th
            #     factor1 = (-w2 * m / (w1 * (u + v))) * (np.exp(w1 * t) * v * w1 - 1 + w1 * t * v / m)
            #     factor2 = np.log(w2 * (m * np.exp(w1 * t) + v) / (u + v))
            #     res[i] = (factor1 + factor2)

            # th = np.log(u / m) / w1
            # if t < th:
            #     res[i] = np.log(1e-4)
            #     # return res
            # else:
            #     sigma = np.exp(w1 * (t - th))
            #     piece1 = m * w2 / w1 / (u + v)
            #     piece2 = -w2*(m * sigma + v * w1 * (t - th)) / w1 / (u + v)
            #     piece3 = (v * w1 + m * w1 * sigma)
            #     res[i] = np.log(w2 / (w1 * (u + v))) + piece1 + piece2 + np.log(piece3)

            th = np.log(u / m) / w1
            if t < th:
                res[i] = np.log(1e-4)
                # return res
            else:
                piece = -w2 * m / ((u + v) * w1) * (np.exp(w1 * t) - np.exp(w1 * th) + w1 * v * (t - th) / m)
                final = (-np.exp(piece) * (-w2 * m / (w1 * (u + v))) * (np.exp(w1 * t) * w1 + v * w1 / m))
                if final <= 0:
                    print("XDDD")
                res[i] = np.log(final)

        return res

    def log_lkl(self, params, life_spans, m_zeros):

        log_pdfs = self.log_pdf(params, life_spans, m_zeros)
        if np.any(np.isinf(log_pdfs)):
            return -np.inf

        return np.sum(log_pdfs)

    def log_prior(self, params):
        w1, w2, u, v = params
        # if 0 < w1 <= 6 and 0 < w2 <= 6 and 0 < u <= 6 and 0 < v <= 30 and u < v:
        if w1 > 0 and w2 > 0 and v > u > 0 and 0 < v <= 30:
            return 0.0
        else:
            return -np.inf

    def log_prob(self, params, life_spans, m_zeros):
        check = self.log_prior(params)

        if not np.isfinite(check):
            return -np.inf
        else:
            return check + self.log_lkl(params, life_spans, m_zeros)

    def params_after_division(self, final_X):
        return np.array([final_X[0] / 2])

    def update_params(self, new_params):
        super().update_params(new_params)
        self.m0 = new_params[0]


class Model2(GenericModel):
    def __init__(self, m0=None, w1=None, w2=None, u=None, v=None):
        fixed_labels = ["w1", "w2", "u", "v"]
        var_labels = ["m0"]
        if (m0 is not None) and (w1 is not None) and (w2 is not None) and (u is not None) and (v is not None):
            self.m0 = m0
            self.w1 = w1
            self.w2 = w2
            self.u = u
            self.v = v
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels,
                             fixed_params=[w1, w2, u, v], variable_params=[m0])
        else:
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels)

    def X(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        param1 = m0 * np.exp(w1 * t)
        ##############################################
        # param2 = m0 * w2 * (np.exp(w1 * t) - 1) / w1
        param2 = m0 * (np.exp(w1 * t) - 1)
        return np.column_stack([param1, param2]).reshape(-1, 2)

    # def t_star(self):
    #     m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
    #     return (1 / w1) * np.log((w1 * u) / (w2 * m0) + 1)

    # def h(self, t):
    #     m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

    #     res = np.ones(shape=t.shape)
    #     th = self.t_star()
    #     mask = t >= th

    #     if th > 0:
    #         t = t - th

    #     res[mask] = (w2 * (self.X(t)[1, :] + v) / (u + v))[mask]
    #     return res

    def S(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

        res = np.ones(shape=t.shape)
        #################################################
        # th = (1 / w1) * np.log((w1 * u) / (w2 * m0) + 1)
        th = (1 / w1) * np.log((u / m0) + 1)
        mask = t >= th

        # if th > 0:
        #     t = t - th
        #################################################
        # factor = - (w2 ** 2 * m0) / (w1 ** 2 * (u + v))
        # term1 = np.exp(w1 * t[mask]) - 1
        # term2 = w1 * t[mask] * ((v * w1) / (w2 * m0) - 1)
        # res[mask] = np.exp(factor * (term1 + term2))
        factor = - (w2 * m0) / (w1 * (u + v))
        term1 = np.exp(w1 * t[mask]) - np.exp(w1 * th)
        term2 = - w1 * (t[mask] - th)
        term3 = (w1 * v / m0) * (t[mask] - th)
        res[mask] = np.exp(factor * (term1 + term2 + term3))
        return res

    def pdf(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        res = np.zeros(shape=len(t))
        #################################################
        # th = (1 / w1) * np.log((w1 * u) / (w2 * m0) + 1)
        th = (1 / w1) * np.log((u / m0) + 1)

        mask = t >= th
        # t = np.where((th > 0) & mask, t - th, t)

        ################################################
        # factor = ((w2 ** 2) * m0) / ((w1 ** 2) * (u + v))
        # term1 = np.exp(w1 * t) - 1
        # term2 = w1 * t * (v * w1 / (w2 * m0) - 1)
        factor = - (w2 * m0) / (w1 * (u + v))
        term1 = np.exp(w1 * t) - np.exp(w1 * th)
        term2 = - w1 * (t - th)
        term3 = (w1 * v / m0) * (t - th)
        S_term = np.exp(factor * (term1 + term2 + term3))

        # h = factor * (np.exp(w1 * t) * w1 + v * w1 / m0)
        #######################################
        # p = w2 * m0 * (np.exp(w1*t) - 1) / w1
        p = m0 * (np.exp(w1 * t) - 1)
        h = w2 * (p + v) / (u + v)

        ##########################################################
        # res[mask] = (np.exp(-factor * (term1 + term2)) * h)[mask]
        res[mask] = (S_term * h)[mask]
        return res

    # def log_pdf(self, params, life_spans, m0s):
    #     w1, w2, u, v = params
    #     t = life_spans

    #     res = np.zeros(shape=t.shape)
    #     th = (1 / w1) * np.log((w1 * u) / (w2 * m0s) + 1)
    #     th = np.where(th < 0, 0, th)
    #     mask = t >= th

    #     t = t - th

    #     # factor1 = -w2 * m0s * (np.exp(w1 * t) * v * w1 - 1 + w1 * t * v / m0s) / (w1 * (u + v))

    #     factor = ((w2 ** 2) *  m0s) / ((w1 ** 2) * (u + v))
    #     term1 = np.exp(w1 * t) - 1
    #     term2 = w1 * t * (v * w1 / (w2 * m0s) - 1)
    #     factor1 = factor * (term1 + term2)

    #     factor2 = np.log(w2 * (w2 * m0s *(np.exp(w1*t) - 1) / w1 + v) / (u + v))
    #     res[mask] = (factor1 + factor2)[mask]
    #     return res

    def log_pdf(self, params, life_spans, m0s):
        # w1, w2, u, v = params
        # for t, m0 in zip(life_spans, m0s):

        w1, w2, u, v = params
        t_star = (1 / w1) * np.log((u / m0s) + 1)  # if m0 is a vector then it's a vector

        out = np.zeros(len(life_spans))
        for i, (s, m) in enumerate(zip(life_spans, m0s)):
            if s < t_star[i]:
                out[i] = np.log(1e-4)
            else:
                ###############################################################
                # factor = -(w2**2 * m) / (w1**2 * (u + v))
                # term1 = np.exp(w1 * (s - t_star[i]))
                # term2 = w1 * (s - t_star[i]) * ((v * w1) / (w2 * m) - 1)
                # term3 = -1
                # # sur = np.exp(factor * (term1 + term2 + term3))
                # sur = factor * (term1 + term2 + term3)
                # p = (w2 * m / w1) * (np.exp(w1 * (s - t_star[i])) - 1)
                # # out[i] = np.log(sur) + np.log(w2) + np.log(p + v) - np.log(u + v)
                # out[i] = sur + np.log(w2) + np.log(p + v) - np.log(u + v)
                # add1 = np.log(w2 /(w1 * (u + v)))
                # factor = - (w2 * m) / (w1 * (u + v))
                # term1 = np.exp(w1 * (s - t_star[i])) - 1
                # term2 = w1 * (s - t_star[i]) * ((v / m) - 1)
                # add2 = factor * (term1 + term2)
                # add3 = np.log(v + m * (np.exp(w1 * (s - t_star[i])) - 1))
                factor = w2 / (w1 * (u + v))
                term1 = - factor * (m * np.exp(s * w1) - m * s * w1 + s * v * w1)
                term2 = factor * (v * w1 * t_star[i] - m * w1 * t_star[i] + u + m)
                term3 = np.log(v * w1 - m * w1 + m * w1 * np.exp(s * w1))
                out[i] = np.log(factor) + term1 + term2 + term3

        return out

    def log_lkl(self, params, life_spans, m_zeros):
        log_pdfs = self.log_pdf(params, life_spans, m_zeros)
        if np.any(np.isinf(log_pdfs)):
            print("XDZZ")
            return -np.inf

        return np.sum(log_pdfs)

    def log_prior(self, params):
        w1, w2, u, v = params
        if w1 > 0 and 10 > w2 > 0 and v > u > 0 and 0 < v <= 40:
            return 0.0
        else:
            return -np.inf

    def log_prob(self, params, life_spans, m_zeros):
        check = self.log_prior(params)

        if not np.isfinite(check):
            return -np.inf
        else:
            return check + self.log_lkl(params, life_spans, m_zeros)

    def params_after_division(self, final_X):
        return np.array([final_X[0] / 2])

    def update_params(self, new_params):
        super().update_params(new_params)
        self.m0 = new_params[0]


class Model3(GenericModel):
    def __init__(self, a=None, b=None, c=None, d=None, m_f=None, w2=None, u=None, v=None, k=None, alpha=None):

        fixed_labels = ["a", "b", "c", "d", "w2", "u", "v"]
        var_labels = ["k", "m_f", "alpha"]

        if ((a is not None) and (b is not None) and (c is not None) and (d is not None) and
                (m_f is not None) and (w2 is not None) and (u is not None) and (v is not None)):

            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.k = k if k is not None else np.random.beta(c, d)
            self.m_f = m_f
            self.alpha = alpha if alpha is not None else np.random.gamma(a, b)
            self.w2 = w2
            self.u = u
            self.v = v
            # super().__init__(fixed_params={"a": a, "b": b, "c": c, "d": d, "w2": w2, "u": u, "v": v},
            #                  variable_params={"k": self.k, "m_f": m_f, "alpha": self.alpha})
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels,
                             fixed_params=[a, b, c, d, w2, u, v],
                             variable_params=[self.k, m_f, self.alpha])
        else:
            # When we want just to use the log_lkl/pdfs we don't need to pass parameters
            super().__init__(fixed_labels=fixed_labels, var_labels=var_labels)

    def X(self, t):
        k, m_f, alpha, w2, u, v = self.k, self.m_f, self.alpha, self.w2, self.u, self.v
        m0 = k * m_f
        param1 = m0 * np.exp(alpha * t)
        param2 = m0 * (np.exp(alpha * t) - 1)
        return np.column_stack([param1, param2]).reshape(-1, 2)

    def S(self, t):
        k, m_f, alpha, w2, u, v = self.k, self.m_f, self.alpha, self.w2, self.u, self.v
        m0 = k * m_f

        res = np.ones(shape=t.shape)
        th = (1 / alpha) * np.log((u / m0) + 1)
        mask = t >= th

        
          
        factor1 = -(w2*m0) / (alpha * (u + v))
        term1 = np.exp(alpha*t)-np.exp(alpha*th)
        factor2 = (w2*m0)/(u+v)
        term2 = t-th
        factor3 = -(w2*v)/(u+v)
        term3 = t-th
        res[mask] = np.exp(factor1 * term1 + factor2*term2 +factor3*term3)[mask]
        return res

    def pdf(self, t, params=None):
        if params is None:
            k, m_f, alpha, w2, u, v = self.k, self.m_f, self.alpha, self.w2, self.u, self.v
        else:
            k, m_f, alpha, w2, u, v = params

        m0 = k * m_f

        res = np.zeros(shape=t.shape)
        th = (1 / alpha) * np.log((u / m0) + 1)

        th = np.where(th < 0, 0, th)
        mask = t >= th

        factor1 = -(w2*m0) / (alpha * (u + v))
        term1 = np.exp(alpha*t)-np.exp(alpha*th)
        factor2 = (w2*m0)/(u+v)
        term2 = t-th
        factor3 = -(w2*v)/(u+v)
        term3 = t-th

        h = (w2/(u+v))*(-m0*np.exp(alpha*t)+m0-v)

        res[mask] = (-np.exp(factor1 * term1 + factor2*term2 +factor3*term3) * h)[mask]
        return res

    def log_pdf(self, params, life_spans, kappas, m_finals, alphas):
        w2, u, v = params[4:]
        t = life_spans

        m0 = kappas * m_finals
        res = np.zeros(shape=t.shape)
        th = (1 / alphas) * np.log((u / m0) + 1)
        th = np.where(th < 0, 0, th)
        mask = t >= th

        #t = t - th

        factor1 = -(w2*m0) / (alphas * (u + v))
        term1 = np.exp(alphas*t)-np.exp(alphas*th)
        factor2 = (w2*m0)/(u+v)
        term2 = t-th
        factor3 = -(w2*v)/(u+v)
        term3 = t-th

        h = (w2/(u+v))*(-m0*np.exp(alphas*t)+m0-v)

        res[mask] = np.log(-np.exp(factor1 * term1 + factor2*term2 +factor3*term3) * h)[mask]
        return res

    def log_prior(self, params):
        a, b, c, d, w2, u, v = params
        # if 0 < w2 <= 10 and 0 < u <= 10 and 0 < v <= 25 and v > u and 0 < c < 20 and 0 < d < 20 and 0 < a < 60 and 0 < b < 1:
        # if 0 < w2  and 0 < u  and 0 < v  and 0 < c and 0 < d and 0 < a and 0 < b:
        if a > 0 and b > 0 and c > 0 and d > 0 and w2 > 0 and 0 < u < v and 5 > v > 0:
            return 0.0
        else:
            return -np.inf

    def log_lkl(self, params, life_spans, kappas, m_finals, alphas):
        a, b, c, d = params[:4]
        out1 = scipy.stats.gamma.pdf(x=alphas, a=a, scale=b)
        out1[out1 == 0] = 1e-300
        # if 0 in out1:
        #     return -np.inf

        out2 = scipy.stats.beta.pdf(x=kappas, a=c, b=d)
        out2[out2 == 0] = 1e-300
        # if 0 in out2:
        #     return -np.inf

        log_pdfs = self.log_pdf(params, life_spans, kappas, m_finals, alphas)
        if np.any(np.isinf(log_pdfs)):
            return -np.inf

        return np.sum(log_pdfs) + np.sum(np.log(out1)) + np.sum(np.log(out2))

    def log_prob(self, params, life_spans, kappas, m_finals, alphas):
        check = self.log_prior(params)

        if not np.isfinite(check):
            return -np.inf
        else:
            return check + self.log_lkl(params, life_spans, kappas, m_finals, alphas)

    def params_after_division(self, final_X):
        new_alpha = np.random.gamma(self.a, self.b)
        new_k = np.random.beta(self.c, self.d)
        return np.array([new_k, final_X[0], new_alpha])

    def update_params(self, new_params):
        super().update_params(new_params)
        self.k, self.m_f, self.alpha = new_params
