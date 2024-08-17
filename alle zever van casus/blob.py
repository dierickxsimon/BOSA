class Blob:
    def __init__(self) -> None:
        self.x : int 
        self.y : int 
        self.w : int
        self.h : int
        self.id : int = None
        self.frame_detected : int = None
        
    @property
    def x_right_top(self):
        return self.x + self.w
    
    @property
    def y_left_bottom(self):
        return self.y + self.h
    

class BlobHandler:
    def __init__(self) -> None:
        self.max_area:int = None
        
    def 