class Position:
    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    # GETTERS:
    def get_lat(self):
        return self._lat

    def get_lon(self):
        return self._lon
