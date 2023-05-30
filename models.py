import numpy as np
import matplotlib.pyplot as plt
import scipy


class GenericModel:
    def __init__(self, params=None):
        if params is not None:
            self.params_dict = params
            self.params_list = list(params.values())

            if not self.are_params_valid():
                print("WARNING PARAMS ARE NOT VALID")

    def X(self, t):
        raise NotImplementedError()

    def are_params_valid(self):
        raise NotImplementedError()

    def t_star(self, params=None):
        raise NotImplementedError()

    def h(self, t, params=None):
        raise NotImplementedError()

    def S(self, t, params=None):
        raise NotImplementedError()

    def pdf(self, t, params=None):
        out = -self.S_deriv(t, params)
        return out

    def log_lkl(self, params, t):
        raise NotImplementedError()

    def log_prior(self, params, life_spans):
        raise NotImplementedError()

    def log_prob(self, params, spans, m_zeros):
        raise NotImplementedError()

    def update_params(self, new_params):
        pass

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

        # fig = plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(t, S, label="S(t)")
        ax.plot(t, pdf, label="PDF")

        ax.set_xlabel("Time")
        ax.legend()
        title = f"{self.__class__.__name__}  " + " ".join([f"{k}={v:.3g}" for k, v in self.params_dict.items()])
        ax.set_title(title, wrap=True)
        plt.tight_layout()


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
    def __init__(self, m0, w1, w2, u, v):
        self.m0 = m0
        self.w1 = w1
        self.w2 = w2
        self.u = u
        self.v = v
        super().__init__({"m0": m0, "w1": w1, "w2": w2, "u": u, "v": v})

    def X(self, t):
        param1 = self.m0 * np.exp(self.w1 * t)
        return param1.reshape(-1, 1)

    def are_params_valid(self):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        return (m0 > 0) and (w1 > 0) and (w2 > 0) and (u > 0) and (v > 0) and (u < v)

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

    def pdf(self, t, params=None):
        if params is None:
            m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        else:
            m0, w1, w2, u, v = params

        res = np.zeros(shape=len(t))
        th = np.log(u / m0) / w1

        mask = t >= th
        t = np.where((th > 0) & mask, t - th, t)

        factor = - (w2 * m0) / (w1 * (u + v))
        term1 = np.exp(w1 * t) - 1
        term2 = w1 * t * v / m0

        # S' = S * h(x)
        return np.inf
        exp_deriv = factor * (np.exp(w1 * t) * w1 + v * w1 / m0)

        res[mask] = -np.exp(factor * (term1 + term2))[mask] * exp_deriv[mask]
        return res

    def params_after_division(self, final_X):
        m0, w1, w2, u, v = self.m0, self.w1, self.w2, self.u, self.v
        return np.array([final_X[0] / 2, w1, w2, u, v])

    def update_params(self, new_params):
        self.m0, self.w1, self.w2, self.u, self.v = new_params
        self.params_dict = {k: new_params[i] for i, k in enumerate(self.params_dict.keys())}
        self.params_list = list(new_params)

    def log_lkl(self, params, ts):

        # res = 0
        # for m0, t in zip(m0s, ts):
        #     res += np.log(self.pdf(t, [m0, w1, w2, u, v]))

        pdfs = self.pdf(ts, params)
        if 0 in pdfs:
            return -np.inf
        # mask = pdfs == 0
        # if np.any(mask):
        #     m0s, w1, w2, u, v = params
        #     # print("Delle pdf hanno tornato zero:")
        #     # for m0, t in zip(m0s[mask], ts[mask]):
        #     #     tstar = np.log(u / m0) / w1
        #     #     print(f"w1={w1:6}, w2={w2:6}, u={u:6}, v={v:6}, m0={m0:6}, t={t:6}, tstar={tstar:6}")
        #     return -np.inf
        return np.sum(np.log(pdfs))

    def log_prior(self, params):
        m0, w1, w2, u, v = params
        if 0 < w1 <= 6 and 0 < w2 <= 6 and 0 < u <= 6 and 0 < v <= 30 and u < v:
            return 0.0
        else:
            return -np.inf

    def log_prob(self, params, life_spans, m_zeros):
        params = list([m_zeros, *params])
        check = self.log_prior(params)
        if not np.isfinite(check):
            return -np.inf
        else:
            value = self.log_lkl(params, life_spans)
            return check + value


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
            super().__init__({"a": a, "b": b, "c": c, "d": d,
                              "k": self.k, "m_f": m_f, "alpha": self.alpha,
                              "w2": w2, "u": u, "v": v})
        else:
            # When we want just to use the log_lkl/pdfs we don't need to pass parameters
            super().__init__(params=None)

    def are_params_valid(self):
        k, m_f, alpha, w2, u, v = self.k, self.m_f, self.alpha, self.w2, self.u, self.v
        return (k > 0) and (m_f > 0) and (alpha > 0) and (w2 > 0) and (u > 0) and (v > 0) and (u < v)

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

    def log_pdf(self, t, params=None):
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

        factor1 = w2 / (alpha * (u + v)) * (m0 * (1 - np.exp(alpha * t)) + alpha * t * (m0 - v))
        factor2 = np.log(w2 * m0 / (u + v)) + np.log(np.exp(alpha * t) - 1 + v / m0)
        res[mask] = (factor1 + factor2)[mask]
        return res

    def log_prior(self, params):
        a, b, c, d, k, m_f, alphas, w2, u, v = params
        # if 0 < w2 <= 10 and 0 < u <= 10 and 0 < v <= 25 and v > u and 0 < c < 20 and 0 < d < 20 and 0 < a < 60 and 0 < b < 1:
        # if 0 < w2  and 0 < u  and 0 < v  and 0 < c and 0 < d and 0 < a and 0 < b:
        if 0 < w2 and 0 < u and 0 < v and v > u and 0 < c and 0 < d and 0 < a and 0 < b and u < 100 and v < 100 and w2 < 100:
            return 0.0
        else:
            return -np.inf

    def log_lkl(self, params, ts):
        a, b, c, d, kappas, m_f, alphas, w2, u, v = params
        out1 = scipy.stats.gamma.pdf(x=alphas, a=a, scale=b)
        out1[out1 == 0] = 1e-300
        # if 0 in out1:
        #     return -np.inf

        out2 = scipy.stats.beta.pdf(x=kappas, a=c, b=d)
        out2[out2 == 0] = 1e-300
        # if 0 in out2:
        #     return -np.inf

        log_pdfs = self.log_pdf(ts, params[4:])
        if np.any(np.isinf(log_pdfs)):
            return -np.inf

        return np.sum(log_pdfs) + np.sum(np.log(out1)) + np.sum(np.log(out2))

    def log_prob(self, params, life_spans, alphas, kappas, m_finals):
        params = list([*params[:4], kappas, m_finals, alphas, *params[4:]])
        check = self.log_prior(params)

        if not np.isfinite(check):
            return -np.inf
        else:
            return check + self.log_lkl(params, life_spans)

    def params_after_division(self, final_X):
        a, b, c, d = self.a, self.b, self.c, self.d
        k, m_f, alpha, w2, u, v = self.k, self.m_f, self.alpha, self.w2, self.u, self.v
        new_alpha = np.random.gamma(a, b)
        new_k = np.random.beta(c, d)
        return np.array([a, b, c, d, new_k, final_X[0], new_alpha, w2, u, v])

    def update_params(self, new_params):
        self.a, self.b, self.c, self.d, self.k, self.m_f, self.alpha, self.w2, self.u, self.v = new_params
        self.params_dict = {k: new_params[i] for i, k in enumerate(self.params_dict.keys())}
        self.params_list = list(new_params)
