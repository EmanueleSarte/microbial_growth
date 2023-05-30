import numpy as np

class Evolver:
    def __init__(self, model):
        self.model = model

        self.time = None
        self.X = None
        self.params = None

        self.div_times = None
        self.born_times = None
        self.spans = None

        self.offset_start = None
        self.offset_end = None

        self._evolved = False

    def get_data(self, start, stop):
        if not self._evolved:
            print("Still not evolved")
            return
        
        start = start or 0
        stop = stop or len(self.spans)
        
        time = self.time[self.offset_start[start] : self.offset_end[stop] + 1]
        data = self.X[self.offset_start[start] : self.offset_end[stop] + 1]
        return time, data
    
    def get_final_data(self, start=None, stop=None):
        if not self._evolved:
            print("Still not evolved")
            return
        start = start or 0
        stop = stop or len(self.spans)

        time = self.time[self.offset_end[start:stop + 1]]
        data = self.X[self.offset_end[start:stop + 1]]
        return time, data
    
    def get_start_data(self, start, stop):
        if not self._evolved:
            print("Still not evolved")
            return
        
        start = start or 0
        stop = stop or len(self.spans)
    
        time = self.time[self.offset_start[start:stop + 1]]
        data = self.X[self.offset_start[start:stop + 1]]
        return time, data
    
    def get_alphas(self):
        if self.model.__class__.__name__.lower() != "model3":
            print("Alphas are availables only on model 3")
            return None
        return self.params[:, 6]
    
    def get_kappas(self):
        if self.model.__class__.__name__.lower() != "model3":
            print("Kappas are availables only on model 3")
            return None
        return self.params[:, 4]
    
    def get_mfinals(self):
        if self.model.__class__.__name__.lower() != "model3":
            print("m finals are availables only on model 3")
            return None
        return self.params[:, 5]


    def evolve(self, n_div, debug=False):
        if self._evolved:
            print("Already evolved")
            return
        self._evolved = True

        # MAX_T = 8
        T_PER_SEC = 1000 

        self.div_times = np.zeros(shape=n_div)
        self.born_times = np.zeros(shape=n_div)
        # self.params = np.zeros(shape=n_div)
        self.offset_start = np.zeros(shape=n_div, dtype=int)
        self.offset_end = np.zeros(shape=n_div, dtype=int)
        self.time = np.array([])
        

        for i in range(n_div):
            if debug:
                print(self.model.params_dict)

            if self.params is None:
                self.params = np.zeros(shape=(n_div, len(self.model.params_list)))
            self.params[i, :] = np.array(self.model.params_list)

            test_range = np.arange(1, 40, 1)
            test_values = self.model.S(t=test_range)
            mask = np.argmax(np.isclose(test_values, 0, atol=1e-5, rtol=0), axis=0)
            max_t = test_range[mask]


            if mask == len(test_range) - 1:
                print("The maximum time range could be incorrect")

            if debug:
                print(self.model.params_dict, max_t, self.model.m0)

            time = np.linspace(0, max_t, max_t * T_PER_SEC)
            probs = self.model.S(t=time)

            r = np.random.uniform(low=0, high=np.max(probs[probs != 1.]))
            index = np.where((probs > r) & (probs != 1))[0][-1]

            X_data = self.model.X(time)
            
            self.offset_start[i] = 0 if i == 0 else self.offset_end[i - 1] + 1
            self.offset_end[i] = self.offset_start[i] + index - 1

            # print(f"Inizio {self.offset_start[i]}, Fine {self.offset_end[i]}, Index: {index}")

            if self.X is None:
                self.X = X_data[:index, :]
            else:
                self.X = np.concatenate((self.X, X_data[:index, :]))

            # if self.X is None:
            #     self.X = np.zeros(shape=(T_PER_SEC * 8 * n_div, X_data.shape[1]))

            self.X[self.offset_start[i] : self.offset_end[i] + 1, :] = X_data[:index, :] 

            step = time[1] - time[0]
            if i == 0:
                self.div_times[i] = time[index - 1]
                self.born_times[i] = 0
                self.time = time[:index]
            else:
                self.div_times[i] = time[index - 1] + step + self.time[-1]
                self.born_times[i] = self.time[-1] + step
                self.time = np.concatenate((self.time, self.time[-1] + time[:index] + step))

            new_params = self.model.params_after_division(X_data[index, :].reshape(-1))
            self.model.update_params(new_params)

        # self.X = self.X[:self.offset_end[-1] + 1, :]
        # print("\n".join(self.X[:, 0].astype(str)))
        self.spans = self.div_times[:-1] - self.born_times[:-1]