import numpy as np
import matplotlib.pyplot as plt
import scipy


class GenericModel:
    def __init__(self, fixed_labels, var_labels, fixed_params=None, variable_params=None):
        self.fix_params_label = fixed_labels.copy()
        self.var_params_label = var_labels.copy()

        if fixed_params:
            #     self.fix_params_label = []
            self.fix_params = fixed_params.copy()
        #     for label, param in fixed_params.items():
        #         self.fix_params_label.append(label)
        #         self.fix_params.append(param)

        if variable_params:
            #     self.var_params_label = []
            self.var_params = variable_params.copy()
        #     for label, param in variable_params.items():
        #         self.var_params_label.append(label)
        #         self.var_params.append(param)

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

        ax.set_xlabel("Time")
        ax.legend()
        title = f"{self.__class__.__name__}  " + self.get_params_str()
        ax.set_title(title, wrap=True)
        plt.tight_layout()

    def get_params_str(self):
        params_list = [a for a in zip(self.fix_params_label, self.fix_params)]
        params_list += [a for a in zip(self.var_params_label, self.var_params_initial)]
        return " ".join([f"{k}={v:.3g}" for k, v in params_list])


class Model1_1:
    def __init__(self, m0, w1, w2, u, v):
        self.initial_m0 = m0

        self.m0 = m0
        self.w1 = w1
        self.w2 = w2
        self.u = u
        self.v = v

    def X(self, t):
        return (self.m0 + self.u) * np.exp(self.w1 * t) - self.u

    def S(self, t, params=None):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

        term1 = w2 * (u/v - 1) * t
        term2 = w2 * (m0 + u)/(w1*v) * (np.exp(w1 * t) - 1)
        res = np.ones(shape=t.shape) * np.exp(term1 - term2)
        return res

    def S_deriv(self, t, params=None, m0=None):
        w1, w2, u, v = params if params is not None else (self.w1, self.w2, self.u, self.v)
        m0 = m0 if m0 is not None else self.m0

        term1 = w2 * (u/v - 1) * t
        term2 = w2 * (m0 + u)/(w1*v) * (np.exp(w1 * t) - 1)
        res = np.exp(term1 - term2) * (w2 * (u/v - 1) - w2 * (m0 + u) * np.exp(w1*t) / v)
        return res

    def pdf(self, t, params=None, m0=None):
        w1, w2, u, v = params if params is not None else (self.w1, self.w2, self.u, self.v)
        m0 = m0 if m0 is not None else self.m0
        term1 = w2 * (u/v - 1) * t
        term2 = w2 * (m0 + u)/(w1*v) * (np.exp(w1 * t) - 1)
        out = -np.exp(term1 - term2) * (w2 * (u/v - 1) - w2 * (m0 + u) * np.exp(w1*t) / v)
        out[out == 0] = 1e-300
        return out

    def params_after_division(self, final_X):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        return np.array([final_X[0] / 2, w1, w2, u, v])

    def update_params(self, new_params):
        self.m0, self.w1, self.w2, self.u, self.v = new_params
        self.params_dict = {k: new_params[i] for i, k in enumerate(self.params_dict.keys())}
        self.params_list = list(new_params)

    def log_lkl(self, params, spans, m_zeros):
        res = np.log(self.pdf(params, spans, m_zeros))
        return np.sum(res)

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
        if th > 0:
            t = t - th

        factor = - (w2 * m0) / (w1 * (u + v))
        term1 = np.exp(w1 * t) - 1
        term2 = w1 * t * v / m0
        res[mask] = np.exp(factor * (term1 + term2))[mask]
        return res

    def pdf(self, t):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        res = np.zeros(shape=len(t))
        th = np.log(u / m0) / w1

        mask = t >= th
        t = np.where((th > 0) & mask, t - th, t)

        factor = (w2 * m0) / (w1 * (u + v))
        term1 = np.exp(w1 * t) - 1
        term2 = w1 * t * v / m0

        h = factor * (np.exp(w1 * t) * w1 + v * w1 / m0)
        # h = factor * np.exp(w1 * t) * v * w1

        res[mask] = (np.exp(-factor * (term1 + term2)) * h)[mask]
        return res

    def log_pdf(self, params, life_spans, m0s):
        w1, w2, u, v = params
        t = life_spans

        res = np.zeros(shape=t.shape)
        th = np.log(u / m0s) / w1
        th = np.where(th < 0, 0, th)
        mask = t >= th

        t = t - th

        factor1 = -w2 * m0s * (np.exp(w1 * t) * v * w1 - 1 + w1 * t * v / m0s) / (w1 * (u + v))
        factor2 = np.log(w2 * (m0s * np.exp(w1 * t) + v) / (u + v))
        res[mask] = (factor1 + factor2)[mask]
        return res

    def log_lkl(self, params, life_spans, m_zeros):

        log_pdfs = self.log_pdf(params, life_spans, m_zeros)
        if np.any(np.isinf(log_pdfs)):
            print("XDZZ")
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


# class Model2(GenericModel):
#     def __init__(self, m0, w1, w2, u, v):
#         self.m0 = m0
#         self.w1 = w1
#         self.w2 = w2
#         self.u = u
#         self.v = v
#         super().__init__({"m0": m0, "w1": w1, "w2": w2, "u": u, "v": v})

#     def are_params_valid(self):
#         m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
#         return (m0 > 0) and (w1 > 0) and (w2 > 0) and (u > 0) and (v > 0) and (u < v)

#     def X(self, t):
#         m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
#         param1 = m0 * np.exp(w1 * t)
#         param2 = m0 * w2 * (np.exp(w1 * t) - 1) / w1
#         return np.array([param1, param2])

#     def t_star(self):
#         m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
#         return (1 / w1) * np.log((w1 * u) / (w2 * m0) + 1)

#     def h(self, t):
#         m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

#         res = np.ones(shape=t.shape)
#         th = self.t_star()
#         mask = t >= th

#         if th > 0:
#             t = t - th

#         res[mask] = (w2 * (self.X(t)[1, :] + v) / (u + v))[mask]
#         return res

#     def S(self, t):
#         m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v

#         res = np.ones(shape=t.shape)
#         th = self.t_star()
#         mask = t >= th

#         if th > 0:
#             t = t - th
#         factor = - (w2 ** 2 * m0) / (w1 ** 2 * (u + v))
#         term1 = np.exp(w1 * t[mask]) - 1
#         term2 = w1 * t[mask] * ((v * w1)/ (w2 * m0) - 1)
#         res[mask] = np.exp(factor * (term1 + term2))
#         return res


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

        if th > 0:
            t = t - th
        factor = w2 / (alpha * (u + v))
        term1 = m0 * (1 - np.exp(alpha * t))
        term2 = alpha * t * (m0 - v)
        res[mask] = np.exp(factor * (term1 + term2))[mask]
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

        t = t - th
        factor = w2 / (alpha * (u + v))
        term1 = m0 * (1 - np.exp(alpha * t))
        term2 = alpha * t * (m0 - v)

        h = w2 * m0 * (np.exp(alpha * t) - 1 + v / m0) / (u + v)

        res[mask] = (np.exp(factor * (term1 + term2)) * h)[mask]
        return res

    def log_pdf(self, params, life_spans, kappas, m_finals, alphas):
        w2, u, v = params[4:]
        t = life_spans

        m0 = kappas * m_finals
        res = np.zeros(shape=t.shape)
        th = (1 / alphas) * np.log((u / m0) + 1)
        th = np.where(th < 0, 0, th)
        mask = t >= th

        t = t - th

        factor1 = w2 / (alphas * (u + v)) * (m0 * (1 - np.exp(alphas * t)) + alphas * t * (m0 - v))
        factor2 = np.log(w2 * m0 / (u + v)) + np.log(np.exp(alphas * t) - 1 + v / m0)
        res[mask] = (factor1 + factor2)[mask]
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
