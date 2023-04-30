class EdgeTravel:
    def __init__(self, train_id, dep_st_id, day, dep_time, travel_time, arr_st_id):
        """
        This constructor creates an EdgeTravel object. An EdgeTravel object is an object that represents a train
        traveling from one station to another, on a specific day, at a specific time.
        :param train_id: The id of the train passing through the edge.
        :param dep_st_id: The id of the station from which the train departs.
        :param day: The day on which the train passes through the edge.
        :param dep_time: The time at which the train passes through the edge.
        :param travel_time: The time it takes for the train to travel from one station to another.
        :param arr_st_id: The id of the station at which the train arrives.
        """
        self._train_id = train_id
        self._dep_st_id = dep_st_id
        self._day = day
        self._dep_time = dep_time
        self._travel_time = travel_time
        self._arr_st_id = arr_st_id

    # GETTERS:
    def get_train_id(self):
        """
        This method returns the id of the train passing through the edge, given an EdgeTravel object.
        :return: The id of the train passing through the edge.
        """
        return self._train_id

    def get_dep_st_id(self):
        """
        This method returns the id of the station from which the train departs, given an EdgeTravel object.
        :return: The id of the station from which the train departs.
        """
        return self._dep_st_id

    def get_day(self):
        """
        This method returns the day on which the train passes through the edge, given an EdgeTravel object.
        :return: The day on which the train passes through the edge.
        """
        return self._day

    def get_dep_time(self):
        """
        This method returns the time at which the train passes through the edge, given an EdgeTravel object.
        :return: The time at which the train passes through the edge.
        """
        return self._dep_time

    def get_travel_time(self):
        """
        This method returns the time it takes for the train to travel from one station to another, given an
        :return: The time it takes for the train to travel from one station to another.
        """
        return self._travel_time

    def get_arr_st_id(self):
        """
        This method returns the id of the station at which the train arrives, given an EdgeTravel object.
        :return: The id of the station at which the train arrives.
        """
        return self._arr_st_id
