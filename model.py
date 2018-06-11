

class Model(object):
    MODEL_POINTS = {
        "A" : [0, 0, 0],
        "C" : [0, 20, 0],
        "F" : [0, 30, 0],
        "B" : [20, 0, 0],
        "D" : [20, 20, 0],
        "G" : [20, 38, 0],
        "E" : [30, 20, 0],
        "H" : [0,0,1]
    }

    # working set, set we are going to use
    DEFAULT_W_S = ["A", "B", "C", "D", "E", "F", "G", "H"]

    """
    @staticmethod
    def normalized_model():
        max_left = 30
        new_mod = {}
        for name, p in Model.MODEL_POINTS.items():
            new_mod[name] = p
            new_mod[name][0] -= max_left
        return new_mod


    @staticmethod
    def point_comparer(p1, p2, order=[1, 0, 2]):
        for i in order:
            if p1[i] != p2[i]:
                return int(p1[i] - p2[i])
        return 0

    # a mapped point is essentialy identical to a
    # regular point, only that it is a tuple that
    # contains the points name as the first arg
    @staticmethod
    def mapped_point_comparer(p1, p2):
        return Model.point_comparer(p1[1], p2[1])

    def __init__(self, p0, w_s = DEFAULT_W_S):
        self._sorted_model = []
        self._norm_model = Model.normalized_model()
        self._w_s = w_s
        self._mapping = {}

        # sort model points

        self._sorted_model = sorted(
            [(key, tuple(self._norm_model[key])) for key in w_s],
            cmp=Model.mapped_point_comparer,
        )

        self._sorted_points = sorted(
            [(i, p.tolist()[0]) for i, p in enumerate(p0)],
            cmp=Model.mapped_point_comparer,
        )

        # print self._sorted_model
        # print self._sorted_points

        # assign initial points to w_s
        for m_p, f_p in zip(self._sorted_model, self._sorted_points):
            self._mapping[f_p[0]] = m_p[0]


    def get_mapping(self):
        return self._mapping

    def update_points(self, p1, st):
        removed_points = 0
        new_mapping = {}

        print self._mapping
        # update points based on optical flow
        for index in range(len(st.tolist())):
            if not st[index]:
                removed_points += 1
            else:
                new_mapping[index - removed_points] = self._mapping[index]

        self._mapping = new_mapping

    def reset_points(self, p0):
        pass
        """

    def tag_points(self, points):
        pass
