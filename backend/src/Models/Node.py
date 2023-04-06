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
