class Node:
    def __init__(self, id, position, passages):
        self._id = id
        self._position = position
        self._passages = passages

    # GETTERS:
    def get_id(self):
        return self._id
    def get_position(self):
        return self._position
    def get_passages(self):
        return self._passages

    # SETTERS:
    def set_id(self, id):
        self._id = id
    def set_position(self, position):
        self._position = position
    def set_passages(self, passages):
        self._passages = passages
