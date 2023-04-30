class Edge:
    def __init__(self, id, fromNode, destNode, mileage, travels):
        """
        This constructor creates an Edge object. The Edge object here represents a connection between two stations, a railway for instance.
        :param id: The id of the edge.
        :param fromNode: The node from which the edge starts.
        :param destNode: The node to which the edge goes.
        :param mileage: The mileage of the edge.
        :param travels: The dictionary of EdgeTravel objects that use this edge.
        """
        self._id = id
        self._fromNode = fromNode
        self._destNode = destNode
        self._mileage = mileage
        self._travels = travels

    # GETTERS:
    def get_id(self):
        """
        This method returns the id of the edge, given an Edge object.
        :return: The id of the edge.
        """
        return self._id

    def get_fromNode(self):
        """
        This method returns the Node from which the edge starts, given an Edge object.
        :return: The Node from which the edge starts.
        """
        return self._fromNode

    def get_destNode(self):
        """
        This method returns the Node to which the edge goes, given an Edge object.
        :return: The Node to which the edge goes.
        """
        return self._destNode

    def get_mileage(self):
        """
        This method returns the mileage of the edge, given an Edge object.
        :return: The mileage of the edge.
        """
        return self._mileage

    def get_travels(self):
        """
        This method returns the dictionary of EdgeTravel objects that use this edge, given an Edge object.
        :return: The dictionary of EdgeTravel objects that use this edge.
        """
        return self._travels
