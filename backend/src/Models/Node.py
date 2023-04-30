class Node:
    def __init__(self, id, position, passages):
        """
        This constructor creates a Node object. The Node object here represents a station.
        :param id: The id of the station.
        :param position: The position of the station.
        :param passages:  The dictionary of Edge objects that use this node.
        """
        self._id = id
        self._position = position
        self._passages = passages

    # GETTERS:
    def get_id(self):
        """
        This method returns the id of the station, given a Node object.
        :return: The id of the station.
        """
        return self._id

    def get_position(self):
        """
        This method returns the position of the station, given a Node object.
        :return: The position of the station.
        """
        return self._position

    def get_passages(self):
        """
        This method returns the dictionary of Edge objects that use this node, given a Node object.
        :return: The dictionary of Edge objects that use this node.
        """
        return self._passages
