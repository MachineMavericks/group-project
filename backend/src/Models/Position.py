class Position:
    def __init__(self, lat, lon):
        """
        This constructor creates a Position object. The Position object here represents a position on the map,
        given by its latitude and longitude.
        :param lat: The latitude of the position, given as a float.
        :param lon: The longitude of the position, given as a float.
        """
        self._lat = lat
        self._lon = lon

    # GETTERS:
    def get_lat(self):
        """
        This method returns the latitude of the position, given a Position object.
        :return: The latitude of the position.
        """
        return self._lat

    def get_lon(self):
        """
        This method returns the longitude of the position, given a Position object.
        :return: The longitude of the position.
        """
        return self._lon
