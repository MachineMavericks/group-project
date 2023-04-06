class Edge:
    def __init__(self, id, fromNode, destNode, mileage, travels):
        self._id = id
        self._fromNode = fromNode
        self._destNode = destNode
        self._mileage = mileage
        self._travels = travels

    # GETTERS:
    def get_id(self):
        return self._id

    def get_fromNode(self):
        return self._fromNode

    def get_destNode(self):
        return self._destNode

    def get_mileage(self):
        return self._mileage

    def get_travels(self):
        return self._travels
