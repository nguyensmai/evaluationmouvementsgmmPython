class Lnode:
    def __init__(self, data, score):
        self.data = data
        self.score = score


class Snode:
    def __init__(self, glabal_, perSegment, perSegmentKP):
        self.global_ = glabal_
        self.perSegment = perSegment
        self.perSegmentKP = perSegmentKP