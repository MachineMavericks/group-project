class NodePassage:
    def __init__(self, train_id, day, arr_time, stay_time):
        self._train_id = train_id
        self._day = day
        self._arr_time = arr_time
        self._stay_time = stay_time

    # GETTERS:
    def get_train_id(self):
        return self._train_id

    def get_day(self):
        return self._day

    def get_arr_time(self):
        return self._arr_time

    def get_stay_time(self):
        return self._stay_time
