class Lnode:
    def __init__(self, data, score):
        self.data = data
        self.score = score


class Snode:
    def __init__(self):
        global_ = None
        perSegment = None
        perSegmentKP = None