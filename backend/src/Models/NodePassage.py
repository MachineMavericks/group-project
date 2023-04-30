class NodePassage:
    def __init__(self, train_id, day, arr_time, stay_time):
        """
        This constructor creates a NodePassage object. A NodePassage object is an object that represents a train
        passing through a station, on a specific day, at a specific time.
        :param train_id: The id of the train passing through the station.
        :param day: The day on which the train passes through the station.
        :param arr_time: The time at which the train passes through the station.
        :param stay_time: The time the train stays at the station.
        """
        self._train_id = train_id
        self._day = day
        self._arr_time = arr_time
        self._stay_time = stay_time

    # GETTERS:
    def get_train_id(self):
        """
        This method returns the id of the train passing through the station, given a NodePassage object.
        :return: The id of the train passing through the station.
        """
        return self._train_id

    def get_day(self):
        """
        This method returns the day on which the train passes through the station, given a NodePassage object.
        :return: The day on which the train passes through the station.
        """
        return self._day

    def get_arr_time(self):
        """
        This method returns the time at which the train passes through the station, given a NodePassage object.
        :return: The time at which the train passes through the station.
        """
        return self._arr_time

    def get_stay_time(self):
        """
        This method returns the time the train stays at the station, given a NodePassage object.
        :return: The time the train stays at the station.
        """
        return self._stay_time
