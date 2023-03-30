class EdgeTravel:
    def __init__(self, train_id, dep_st_id, day, dep_time, travel_time, arr_st_id):
        self._train_id = train_id
        self._dep_st_id = dep_st_id
        self._day = day
        self._dep_time = dep_time
        self._travel_time = travel_time
        self._arr_st_id = arr_st_id

    # GETTERS:
    def get_train_id(self):
        return self._train_id
    def get_dep_st_id(self):
        return self._dep_st_id
    def get_day(self):
        return self._day
    def get_dep_time(self):
        return self._dep_time
    def get_travel_time(self):
        return self._travel_time
    def get_arr_st_id(self):
        return self._arr_st_id

    # SETTERS:
    def set_train_id(self, train_id):
        self._train_id = train_id
    def set_dep_st_id(self, dep_st_id):
        self._dep_st_id = dep_st_id
    def set_day(self, day):
        self._day = day
    def set_dep_time(self, dep_time):
        self._dep_time = dep_time
    def set_travel_time(self, travel_time):
        self._travel_time = travel_time
    def set_arr_st_id(self, arr_st_id):
        self._arr_st_id = arr_st_id